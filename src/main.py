from lib import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('..')

from module import *
from model import *
from dataset import *
from callback import *
from perturb_eval import *

from utils import (
    check_argument_setting,
    get_scheduler_info,
)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import wandb
wandb.login()

@dataclass
class ScriptArguments:
    do_train: bool = field(
        default=False, 
        metadata={"help": "do training if set to true. "}
    )
    do_eval: bool = field(
        default=False, 
        metadata={"help": "do evaluation if set to true. "}
    )
    fp16: bool = field(
        default=False, 
        metadata={"help": ""}
    )
    gpus: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    n_tpu_cores: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    fp16_opt_level: Optional[str] = field(
        default="O2", 
        metadata={"help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "}
    )
    num_train_epochs: Optional[int] = field(
        default=5, 
        metadata={"help": ""}
    )
    max_train_steps: Optional[int] = field(
        default=400, 
        metadata={"help": ""}
    )
    max_eval_steps: Optional[int] = field(
        default=400, 
        metadata={"help": ""}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, 
        metadata={"help": ""}
    )
    gradient_clip_val: Optional[float] = field(
        default=1.0, 
        metadata={"help": ""}
    )
    early_stopping_patience: Optional[int] = field(
        default=-1, 
        metadata={"help": "-1 means never early stop. "
        "early_stopping_patience is measured in validation checks, not epochs. "
        "So val_check_interval will effect it."}
    )
    tgwv_local_ckpt: Optional[str] = field(
        default=None, 
        metadata={"help": "use model from checkpoints. "}
    )
    tgwv_v_mode: Optional[str] = field(
        default=None, 
        metadata={"help": "options: `softmax`, `raw`, `drop`. "}
    )


def get_callback(args):
    if args.main.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.main.early_stopping_patience)
    else:
        es_callback = False

    # set callbacks
    # logging_callback=Seq2SeqLoggingCallback()
    # checkpoint_callback=get_checkpoint_callback(output_dir, val_metric, args.main.save_top_k, lower_is_better)
    early_stopping_callback=es_callback

    checkpoint_callback = None
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            output_dir, monitor="val_{}".format(args.module.val_metric), mode="min", save_top_k=1
        )

def train_settings(args, seed=None, from_argparse=False):
    train_params = {}
    pl.seed_everything(seed)
    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        if from_argparse:
            train_params["amp_level"] = args.fp16_opt_level
            train_params['amp_backend'] = 'apex'

    #if args.main.gpus > 1:
    #    train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.gradient_accumulation_steps
    # train_params['progress_bar_refresh_rate'] = 0

    print('the max number of epochs is {}'.format(args.num_train_epochs))

    return train_params

def trainer_kwargs(script_args: Namespace, model_mode=None, logger=None, callbacks=None) -> Dict:
    assert model_mode in ['seq2seq', 'cls', 'tgwv']
    if model_mode == 'seq2seq':
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            #max_steps=100,
            #min_steps=100,
            max_epochs=script_args.num_train_epochs, 
            min_epochs=script_args.num_train_epochs, 
            check_val_every_n_epoch=1,
            # log_every_n_steps=20,
            gradient_clip_val=script_args.gradient_clip_val,
            # limit_train_batches=1.0,
            # limit_val_batches=1.0,
        )
    elif model_mode == 'cls': # classification
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            #max_steps=100,
            #min_steps=100,
            max_epochs=script_args.num_train_epochs, 
            min_epochs=script_args.num_train_epochs, 
            check_val_every_n_epoch=1,
            # log_every_n_steps=20,
            gradient_clip_val=script_args.gradient_clip_val,
            # limit_train_batches=1.0,
            # limit_val_batches=1.0,
        )
    else: # tgwv
        kwargs = dict(
            accelerator="gpu",
            num_nodes=1,
            logger=logger,
            callbacks=callbacks,
            #max_steps=100,
            #min_steps=100,
            max_epochs=script_args.num_train_epochs, 
            min_epochs=script_args.num_train_epochs, 
            check_val_every_n_epoch=1,
            # log_every_n_steps=20,
            gradient_clip_val=script_args.gradient_clip_val,
            # limit_train_batches=1.0,
            # limit_val_batches=1.0,
        )
    return kwargs

def get_trainer(args, output_dir, logger_name, seed=None, model_mode=None):
    #logger = LoggingCallback.get_logger(logger_name, output_dir)
    #callbacks = get_callback(module, new_args, dict(mode="max", save_top_k=new_args.main.save_top_k))
    return pl.Trainer(
        **trainer_kwargs(
            args, 
            model_mode=model_mode, 
            logger=None, 
            callbacks=None, 
        ), 
        **train_settings(args, seed=seed),
    )

def get_kwargs(
        dataset_len: int, 
        data_args, 
        model_args, 
        loader_args, 
        script_args, 
        model_mode=None,
        cls_model_name=None, 
        v_mode='default', 
        type=None
    ):
    assert type in ['extra_module_attrs', 'output_dir_kwargs']
    assert model_mode in ['seq2seq', 'cls', 'tgwv']
    if model_mode == 'tgwv':
        d_model_mode, m_model_mode = 'cls', 'seq2seq'
    else: 
        d_model_mode = m_model_mode = model_mode
    if type == 'extra_module_attrs':
        kwargs = dict(
            gpus=script_args.gpus,
            sortish_sampler=getattr(data_args, d_model_mode+'_sortish_sampler'), 
            max_tokens_per_batch=loader_args.max_tokens_per_batch, 
            train_batch_size=loader_args.train_batch_size, 
            gradient_accumulation_steps=script_args.gradient_accumulation_steps, 
            dataset_len=dataset_len, 
            num_train_epochs=script_args.num_train_epochs, 
        )
    elif type == 'output_dir_kwargs':
        data_dir = getattr(data_args, d_model_mode+'_data_dir').replace('/', '#')
        model_name = getattr(model_args, m_model_mode+'_model_name_or_path').replace('/', '.')
        if cls_model_name is not None: cls_model_name = cls_model_name.replace('/', '.')
        tgwv_kwargs = {'cls_m': cls_model_name, "v_mode": v_mode} if model_mode == 'tgwv' else {}
        kwargs = dict(
            t=model_mode, # task
            m=model_name, # model
            d=data_dir, # data
            **tgwv_kwargs, 
        )

    return kwargs

def main():
    for data_dir in [Path('../data'), Path('../models')]:
        os.makedirs(data_dir, exist_ok=True)

    parser = HfArgumentParser((
        ClassificationModel.add_specific_args(),
        Seq2SeqDataset.add_specific_args(), 
        ClassificationDataset.add_specific_args(), 
        DataModule.add_specific_args(), 
        TrainingModule.add_specific_args(), 
        *TextGenerationWithVectorInputModel.add_specific_args(), 
        LoggingCallback.add_specific_args(), 
        CheckpointCallback.add_specific_args(), 
        ScriptArguments, 
    ))
    cls_args, seq_data_args, cls_data_args, loader_args, train_args, seq2seq_args, \
        tgwv_args, logging_args, ckpt_args, script_args = parser.parse_args_into_dataclasses()
    args_dict = dict(
        seq2seq=seq2seq_args, 
        cls=cls_args, 
        seq_data=seq_data_args, 
        cls_data=cls_data_args, 
        loader=loader_args, 
        train=train_args, 
        tgwv=tgwv_args, 
        logging=logging_args, 
        ckpt=ckpt_args, 
        script=script_args, 
    )
    for key, value in args_dict.items():
        args_dict[key] = argparse.Namespace(**value.__dict__)

    model = Seq2SeqModel(seq2seq_args)
    cls_model = ClassificationModel(cls_args)
    tgwv_model = TextGenerationWithVectorInputModel(seq2seq_args, tgwv_args)
    
    d_loader = DataModule(loader_args, seq_data_args, 'seq2seq', tokenizer=model.tokenizer)
    cls_loader = DataModule(loader_args, cls_data_args, 'classification', tokenizer=cls_model.tokenizer)

    tgwv_kwargs = {
        'seq_tokenizer':model.tokenizer, 
        'cls_tokenizer':cls_model.tokenizer, 
        'cls_model':cls_model, 
        'padding_num':tgwv_args.d_model - tgwv_model.original_embed_dim - cls_model.args.num_labels, 
        'v_mode': script_args.tgwv_v_mode, 
    }
    tgwv_loader = DataModule(loader_args, cls_data_args, 'tgwv', **tgwv_kwargs)

    seq_list_args = [len(d_loader.get_dataset('train')), seq_data_args, seq2seq_args, ]
    cls_list_args = [len(cls_loader.get_dataset('train')), cls_data_args, cls_args, ]
    tgwv_list_args = [len(tgwv_loader.get_dataset('train')), cls_data_args, seq2seq_args, ]
    common_args = [loader_args, script_args, ]
    seq_attr_kwargs = get_kwargs(*seq_list_args, *common_args, model_mode='seq2seq', type='extra_module_attrs')
    seq_output_dir_kwargs = get_kwargs(*seq_list_args, *common_args, model_mode='seq2seq', type='output_dir_kwargs')
    cls_attr_kwargs = get_kwargs(*cls_list_args, *common_args, model_mode='cls', type='extra_module_attrs')
    cls_output_dir_kwargs = get_kwargs(*cls_list_args, *common_args, model_mode='cls', type='output_dir_kwargs')
    # set tgwv
    tgwv_attr_kwargs = get_kwargs(*tgwv_list_args, *common_args, model_mode='tgwv', type='extra_module_attrs')
    tgwv_output_dir_kwargs = get_kwargs(
        *tgwv_list_args, *common_args, 
        model_mode='tgwv', 
        cls_model_name=cls_model.args.model_name_or_path, 
        v_mode=script_args.tgwv_v_mode, 
        type='output_dir_kwargs'
    )
    """ move to top
    args_dict = dict(
        seq2seq=seq2seq_args, 
        cls=cls_args, 
        seq_data=seq_data_args, 
        cls_data=cls_data_args, 
        loader=loader_args, 
        train=train_args, 
        tgwv=tgwv_args, 
        script=script_args, 
    )
    for key, value in args_dict.items():
        args_dict[key] = argparse.Namespace(**value.__dict__)
    """
    print('start init seq module ...')
    callback_args_dict = dict(log=args_dict['logging'], ckpt=args_dict['ckpt'])
    seq_module = TrainingModule(
        hparams=args_dict['train'], 
        model=model, 
        callback_args_dict=callback_args_dict, 
        output_dir_kwargs=seq_output_dir_kwargs, 
        **seq_attr_kwargs
    )
    print('start init cls module ...')
    cls_module = TrainingModule(
        hparams=args_dict['train'], 
        model=cls_model, 
        callback_args_dict=callback_args_dict, 
        output_dir_kwargs=cls_output_dir_kwargs, 
        **cls_attr_kwargs
    )
    print('start init tgwv module ...')
    # set tgwv
    if script_args.tgwv_local_ckpt is None:
        tgwv_module = TrainingModule(
            hparams=args_dict['train'], 
            model=tgwv_model, 
            callback_args_dict=callback_args_dict, 
            output_dir_kwargs=tgwv_output_dir_kwargs, 
            **tgwv_attr_kwargs
        )
    else: # load ckpt
        tgwv_module = TrainingModule.load_from_checkpoint(
            script_args.tgwv_local_ckpt, 
            model=tgwv_model, 
            callback_args_dict=callback_args_dict, 
            output_dir_kwargs=tgwv_output_dir_kwargs, 
            #**tgwv_attr_kwargs
        )
        print(f'load from {script_args.tgwv_local_ckpt}, success! ')
    """
    cls_trainer = get_trainer(
        script_args, cls_module.output_dir, logging_args.logger_name, 
        seed=cls_module.hparams.seed, 
        model_mode='cls'
    )
    """
    tgwv_trainer = get_trainer(
        script_args, tgwv_module.output_dir, logging_args.logger_name, 
        seed=tgwv_module.hparams.seed, 
        model_mode='tgwv', 
    )
    

    if script_args.do_train:
        tgwv_trainer.fit(tgwv_module, datamodule=tgwv_loader)
    if script_args.do_eval:
        result = tgwv_trainer.test(tgwv_module, datamodule=tgwv_loader)

    import pdb 
    pdb.set_trace()

    # do perturb
    ptb_eval = PerturbEval(
        tgwv_model, cls_model.tokenizer, cls_model.model, 
        tgwv_loader.get_dataloader(type_path='val', shuffle=True), 
        embed_type='unify', 
        perturb_range=(0, 2), 
        use_wandb=True, 
        output_dir=tgwv_module.output_dir, 
        max_source_length=tgwv_loader.dataset_args.max_source_length, 
        ptb_mode=script_args.tgwv_v_mode, 
    )
    ptb_eval.eval_loop()

    import pdb
    pdb.set_trace()

    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.main.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    ######## evaluate ############

    # test() without a model tests using the best checkpoint automatically
    trainer.test()

    if args.main.do_eval:
        Path(args.main.output_dir).mkdir(exist_ok=True)
        if len(os.listdir(args.main.output_dir)) > 3 and args.main.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.main.output_dir))

        victim_model = ModelTrainTemplate(model, args.main.model_mode)

        # print(model)
        dataset = Path(args.main.data_dir).name

        with torch.no_grad():
            model.eval()
            print(dataset)
            model = model.cuda()
            print(model.device)
            data_loader = model.test_dataloader()
            out_lst = []
            for batch_idx, batch in enumerate(data_loader):
                # print(batch)
                batch = model.transfer_batch_to_device(batch, model.device)
                # if batch_idx>10:
                #     continue
                # print(batch['input_ids'].device, model.device)
                out = model.test_step(batch, batch_idx)
                out_lst.append(out)
                print(out['preds'])
                # batch = model.transfer_batch_to_device(batch, 'cpu')
            result = model.test_epoch_end(out_lst)

        for k, v in result.items():
            if k != 'preds':
                print(k, v)

        out_1 = args.main.model_name_or_path
        out_path = os.path.join(out_1, 'test_beam_{}'.format(args.main.length_penalty))
        print('writing the test results to ', out_path)
        with open(out_path, 'w') as f:
            for preds in result['preds']:
                print(preds, file=f)

        # print(result)
        for k, v in result.items():
            if k != 'preds':
                print(k, v)
        


if __name__ == '__main__':
    main()
