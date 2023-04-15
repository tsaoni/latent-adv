from pytorch_lightning.loggers import WandbLogger

logger1 = WandbLogger(project='test', name='run1')
logger2 = WandbLogger(project='test', name='run2')

data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data, columns = ["x", "y"])
wandb.log_metric({"my_custom_plot_id" : wandb.plots.scatter(table, "x", "y", title="Custom Y vs X Scatter Plot")})