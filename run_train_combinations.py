import subprocess

# 定义参数选项
sex_loss_functions = ["CE"]
age_loss_functions = [ "Focal", "LogitAdjust"]
#"LDAM",
# 循环遍历所有参数组合
for sex_loss in sex_loss_functions:
    for age_loss in age_loss_functions:
        print(f"Running training with sex_loss={sex_loss} and age_loss={age_loss}")
        # 构建命令
        command = f"python train.py --sex_loss_function {sex_loss} --age_loss_function {age_loss}"
        # 执行命令
        subprocess.run(command, shell=True)
