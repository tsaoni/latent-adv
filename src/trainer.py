from lib import *
from dataset import *

import wandb
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    logger = get_logger(__name__)
    def __init__(
        self, 
        attacker, eval_model, target_model, 
        train_dataloader, eval_dataloader, 
        weight_decay=1e-4, 
        learning_rate=1e-3, 
        gradient_accumulation_steps=1, 
        lr_scheduler_type="linear", 
        gpus=1, 
        train_batch_size=5, 
        val_batch_size=5, 
        num_train_epochs=1, 
        seed=112, 
        tokenizer_name=None, 
        max_source_length=128, 
        num_warmup_steps=10, 
        output_dir=None, 
        use_wandb=True, 
        n_val_batch=3, 
    ):
        self.config = Namespace(
            weight_decay=weight_decay, 
            learning_rate=learning_rate, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            gpus=gpus, 
            train_batch_size=train_batch_size, 
            val_batch_size=val_batch_size, 
            num_train_epochs=num_train_epochs, 
            seed=seed, 
            tokenizer_name=tokenizer_name, 
            max_source_length=max_source_length, 
            lr_scheduler_type=lr_scheduler_type, 
            num_warmup_steps=num_warmup_steps, 
            resume_from_checkpoint=None, 
            output_dir=output_dir, 
            use_wandb=use_wandb, 
            n_val_batch=n_val_batch, 
            length_penalty=0.1,
            num_beams=4, 
            eval_min_gen_length=6, 
            eval_max_gen_length=50, 
        )

        os.makedirs(output_dir, exist_ok=True)
        self.attacker, self.eval_model, self.target_model = attacker, eval_model, target_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataloaders = {"train": train_dataloader, "val": eval_dataloader}
        self.dataset_len = {k: len(v.dataset) for k, v in self.dataloaders.items()}


    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.config.gpus)  
        effective_batch_size = self.config.train_batch_size * self.config.gradient_accumulation_steps * num_devices
        dataset_size = self.dataset_len["train"]
        return int(dataset_size / effective_batch_size) * self.config.num_train_epochs

    def get_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.configure_optimizers(self.attacker),
            num_warmup_steps=self.config.num_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.total_steps, 
        )
        self.lr_scheduler = lr_scheduler
        return lr_scheduler

    def configure_optimizers(self, model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        return self.optimizer

    def train(self):
        train_dataloader, val_dataloader = (self.dataloaders[k] for k in ["train", "val"])
        lr_scheduler = self.get_lr_scheduler()
        ce_criterion = nn.CrossEntropyLoss()

        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        # Prepare everything with our `accelerator`.
        attacker, eval_model, target_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.attacker, self.eval_model, self.target_model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        self.total_batch_size = self.config.train_batch_size * accelerator.num_processes * self.config.gradient_accumulation_steps
        # Only show the progress bar once on each machine.
        #progress_bar = tqdm(range(self.total_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0
        pad_token_id = self.tokenizer.pad_token_id

        #accelerator.load_state(self.config.resume_from_checkpoint)
        if self.config.use_wandb:
            project_name = os.environ.get('WANDB_PROJECT')
            run = self.config.output_dir.split("/")[-1]
            wandb.init(project=project_name, name=run)

        #print("start evluate target model original accuracy...")
        #self.eval(dataloader=train_dataloader, type="target_origin")
        print("start training... ")
        for epoch in range(starting_epoch, self.config.num_train_epochs):
            attacker.train()
            eval_model.eval()
            target_model.eval()
            progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(train_dataloader):
                run_input_ce_loss, run_target_ce_loss, run_cls_loss, run_acc, run_total_loss = 0, 0, 0, 0, 0
                with accelerator.accumulate(attacker):
                    # prepare batch for attacker
                    data = {k: v for k, v in batch.items() if k not in ["premise", "hypothesis", "labels"]}
                    output = attacker(**data)
                    logits = output.logits
                    # apply gumbel softmax on logits
                    gumbel_logits = F.gumbel_softmax(logits, tau=.1, hard=False)
                    attack_pred_ids = torch.argmax(gumbel_logits, dim=-1).detach()
                    attention_mask = (attack_pred_ids == pad_token_id).to(int)
                    # eval model
                    eval_input_embeds = torch.matmul(gumbel_logits, eval_model.get_input_embeddings().weight)
                    data = {"inputs_embeds": eval_input_embeds, "attention_mask": attention_mask, "decoder_inputs_embeds": eval_input_embeds, "labels": attack_pred_ids}
                    target_ce_loss = eval_model(**data).loss
                    data = {"inputs_embeds": eval_input_embeds, "attention_mask": attention_mask, "decoder_inputs_embeds": eval_input_embeds, "labels": batch["input_ids"]}
                    input_ce_loss = eval_model(**data).loss
                    # target model
                    target_input_embeds = torch.matmul(gumbel_logits, target_model.get_input_embeddings().weight)
                    data = {"inputs_embeds": target_input_embeds, "attention_mask": attention_mask, "decoder_inputs_embeds": target_input_embeds}
                    hidden_states = target_model.model(**data)[0] # since the classification model doesn't support embed input
                    eos_mask = attack_pred_ids.eq(target_model.config.eos_token_id).to(hidden_states.device)
                    row_idxs, col_idxs = torch.where(eos_mask)
                    idx = []
                    unique_row_idxs = row_idxs.unique()
                    for i in range(hidden_states.shape[0]):
                        if i not in unique_row_idxs:
                            idx.append([[hidden_states.shape[1] - 1] * hidden_states.shape[-1]])
                        else:
                            ridxs = torch.where(row_idxs == i)[0]
                            cidx = ridxs[2] if len(ridxs) >= 3 else ridxs[-1]
                            idx.append([[col_idxs[cidx].item()] * hidden_states.shape[-1]])
                    sentence_representation = hidden_states.gather(1, torch.tensor(idx).to(hidden_states.device)).squeeze()
                    preds = target_model.classification_head(sentence_representation)

                    acc = torch.sum(torch.eq(torch.argmax(preds, dim=-1), batch["labels"])).item()
                    cls_loss = ce_criterion(preds, batch["labels"])
                    loss = input_ce_loss #+ target_ce_loss - cls_loss

                    run_total_loss += loss.detach().float()
                    run_input_ce_loss += input_ce_loss.detach().float()
                    run_target_ce_loss += target_ce_loss.detach().float()
                    run_cls_loss += cls_loss.detach().float()
                    run_acc += acc

                    run_input_ce_loss /= self.config.train_batch_size
                    run_target_ce_loss /= self.config.train_batch_size
                    run_cls_loss /= self.config.train_batch_size
                    run_acc /= self.config.train_batch_size
                    run_total_loss /= self.config.train_batch_size

                    if self.config.use_wandb:
                        wandb.log({
                            "train/source_ce": run_input_ce_loss, 
                            "train/target_ce": run_target_ce_loss, 
                            "train/cls_loss": run_cls_loss, 
                            "train/cls_acc": run_acc, 
                            "train/tot_loss": run_total_loss, 
                        })

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                print("Current memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                print("Peak memory allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

            output_dir = f"epoch_{epoch}"
            if self.config.output_dir is not None:
                output_dir = os.path.join(self.config.output_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
            self.eval(dataloader=train_dataloader, output_dir=output_dir, type="attacker_output")


            if epoch < self.config.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(attacker)
                unwrapped_model.save_pretrained(
                    self.config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.config.output_dir)

            accelerator.save_state(output_dir)

        #accelerator.end_training()

        if self.config.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(attacker)
            unwrapped_model.save_pretrained(
                self.config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.config.output_dir)
            #    with open(os.path.join(self.config.output_dir, "all_results.json"), "w") as f:
            #        json.dump({"perplexity": perplexity}, f)

    def eval(self, dataloader=None, output_dir=None, type="target_origin", eval_type="train"):
        if type == "target_origin":
            n_test = 100
            batch_size = self.config.train_batch_size if eval_type == "train" else self.config.val_batch_size
            val_dataloader = self.dataloaders[eval_type] if dataloader is None else dataloader
            progress_bar = tqdm(range(n_test))
            self.target_model.eval()
            val_acc, val_loss = 0., 0.
            #for step, batch in enumerate(val_dataloader):
            for step in range(n_test):
                if step == n_test: break 
                batch = next(iter(val_dataloader))
                data = {k: v for k, v in batch.items() if k not in ["premise", "hypothesis"]}
                output = self.target_model(**data)
                val_loss += output.loss.detach().item()
                preds = torch.argmax(output.logits.detach(), dim=-1)
                val_acc += torch.sum(torch.eq(preds, batch["labels"])).detach().item()
                progress_bar.update(1)
            val_loss /= (n_test * batch_size)
            val_acc /= (n_test * batch_size)
            print(f"the origin target model loss: {val_loss:.2f}, accuracy: {val_acc:.2f}. ")
            if self.config.use_wandb:
                wandb.log({
                    "target/origin_acc": val_acc, 
                    "target/origin_loss": val_loss, 
                })

        elif type == "attacker_output":
            val_dataloader = self.dataloaders[eval_type] if dataloader is None else dataloader
            progress_bar = tqdm(range(self.config.n_val_batch))
            self.attacker.eval()
            decoder_start_token_id = self.attacker.config.decoder_start_token_id
            adv_text_list = []
            for i in range(self.config.n_val_batch):
                batch = next(iter(val_dataloader))
                generated_ids = self.attacker.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True,
                    length_penalty=self.config.length_penalty,
                    decoder_start_token_id=decoder_start_token_id, 
                    num_beams=self.config.num_beams,
                    min_length=self.config.eval_min_gen_length,
                    max_length=self.config.eval_max_gen_length,
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                lmap = lambda f, x: list(map(f, x))
                texts = lmap(str.strip, gen_text)
                for text, premise, hypothesis in zip(texts, batch["premise"], batch["hypothesis"]):
                    result = {"generated": text, "premise": premise, "hypothesis": hypothesis}
                    adv_text_list.append(result)
                progress_bar.update(1)

            with open(os.path.join(output_dir, "adv_texts.json"), "w") as f:
                json.dump(adv_text_list, f, indent=4)
