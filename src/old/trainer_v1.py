from lib import *
from dataset import *

class mlmTrainer:
    logger = get_logger(__name__)
    def __init__(
        self, 
        model, 
        weight_decay=1e-4, 
        learning_rate=1e-3, 
        gradient_accumulation_steps=1, 
        lr_scheduler_type="linear", 
        gpus=1, 
        train_batch_size=5, 
        val_batch_size=5, 
        num_train_epochs=1, 
        data_dir=None, 
        sampler_type="random", 
        dataloader_drop_last=False, 
        num_workers=1, 
        seed=112, 
        tokenizer_name=None, 
        max_source_length=128, 
        num_warmup_steps=10, 
        checkpointing_steps=50, 
        output_dir=None, 
    ):
        self.config = Namespace(
            weight_decay=weight_decay, 
            learning_rate=learning_rate, 
            gradient_accumulation_steps=gradient_accumulation_steps, 
            gpus=gpus, 
            train_batch_size=train_batch_size, 
            val_batch_size=val_batch_size, 
            num_train_epochs=num_train_epochs, 
            data_dir=data_dir, 
            sampler_type=sampler_type, 
            dataloader_drop_last=dataloader_drop_last, 
            num_workers=num_workers, 
            seed=seed, 
            tokenizer_name=tokenizer_name, 
            max_source_length=max_source_length, 
            lr_scheduler_type=lr_scheduler_type, 
            num_warmup_steps=num_warmup_steps, 
            resume_from_checkpoint=None, 
            checkpointing_steps=checkpointing_steps, 
            output_dir=output_dir, 
        )
        # run time attribute
        os.makedirs(output_dir, exist_ok=True)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_len = {k: len(self.get_dataset(type_path=k)) for k in ["train", "val", "test"]}

    def get_dataset(self, type_path="train", ds_class="seq2seq"):
        ds_class_dict = {
            "mlm": mlm_dataset, 
            "seq2seq": Seq2SeqDataset, 
        }
        dataset_class = ds_class_dict[ds_class]
        if ds_class == "mlm":
            path = os.path.join(self.config.data_dir, f"{type_path}.txt")
            kwargs_name = ["tokenizer_name", "mlm", "mlm_probability"]
            c_kwargs = get_kwargs(self.config, kwargs_name)
            c_kwargs.update({"block_size": self.config.max_source_length})
        else: # seq2seq
            path = self.config.data_dir
            kwargs_name = ["tokenizer_name"]
            c_kwargs = get_kwargs(self.config, kwargs_name)
            c_kwargs.update({"type_path": type_path})
        return dataset_class(
            path, 
            **c_kwargs, 
        )

    def get_dataloader(self, type_path="train", ds_class="seq2seq"):
        kwargs_name = ["sampler_type", "dataloader_drop_last", "num_workers", "seed", "tokenizer_name", 
                    "mlm", "mlm_probability", ]
          
        c_kwargs = get_kwargs(self.config, kwargs_name)
        return get_dataloader(
            self.config.data_dir, 
            type_path=type_path, 
            batch_size=self.config.train_batch_size, 
            block_size=self.config.max_source_length, 
            **c_kwargs, 
        )

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.config.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.config.train_batch_size * self.config.gradient_accumulation_steps * num_devices
        dataset_size = self.dataset_len["train"]
        return int(dataset_size / effective_batch_size) * self.config.num_train_epochs

    def get_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.configure_optimizers(),
            num_warmup_steps=self.config.num_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.total_steps, 
        )
        self.lr_scheduler = lr_scheduler
        return lr_scheduler

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        return self.optimizer

    def train(self):
        lr_scheduler = self.get_lr_scheduler()
        train_dataloader = self.get_dataloader(type_path="train")
        val_dataloader = self.get_dataloader(type_path="val")
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        self.total_batch_size = self.config.train_batch_size * accelerator.num_processes * self.config.gradient_accumulation_steps
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.total_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.config.resume_from_checkpoint:
            if self.config.resume_from_checkpoint is not None or self.config.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {self.config.resume_from_checkpoint}")
                accelerator.load_state(self.config.resume_from_checkpoint)
                path = os.path.basename(self.config.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.config.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        num_update_steps_per_epoch = int(self.total_steps / self.config.num_train_epochs)
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch

        self.log_info("train")
        for epoch in range(starting_epoch, self.config.num_train_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if self.config.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % self.config.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                   
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(self.config.checkpointing_steps, int):
                    if completed_steps % self.config.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if self.config.output_dir is not None:
                            output_dir = os.path.join(self.config.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= self.total_steps: # can automatically setting
                    break

            model.eval()
            losses = []
            for step, batch in enumerate(val_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(self.config.val_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            self.logger.info(f"epoch {epoch}: perplexity: {perplexity}")

            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

            if epoch < self.config.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    self.config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.config.output_dir)

            output_dir = f"epoch_{epoch}"
            if self.config.output_dir is not None:
                output_dir = os.path.join(self.config.output_dir, output_dir)
            accelerator.save_state(output_dir)

        accelerator.end_training()

        if self.config.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                self.config.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.config.output_dir)
                with open(os.path.join(self.config.output_dir, "all_results.json"), "w") as f:
                    json.dump({"perplexity": perplexity}, f)

    def eval(self):
        return
        lr_scheduler = self.get_lr_scheduler()
        accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps)
        # Prepare everything with our `accelerator`.
        model, optimizer, eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, eval_dataloader, lr_scheduler
        )


    def test(self):
        val_dataloader = self.get_dataloader(type_path="val")
        for step, batch in enumerate(val_dataloader):
            texts = self.model.generate(batch, return_ids=True, tokenizer=self.tokenizer)
            import pdb 
            pdb.set_trace()


    def log_info(self, msg_type: str):
        if msg_type == "train":
            self.logger.info("***** Running training *****")
            self.logger.info(f"  Num examples = {self.dataset_len['train']}")
            self.logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
            self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
            self.logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {self.total_steps}")