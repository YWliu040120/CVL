use the environment same as da2 is ok!

## 📊 Evaluation

To assess the performance and generalization capability of a trained model, execute the following command:

```bash
python test.py --split samearea --checkpoint ./runs/checkpoints/dpt_vigor_samearea_epoch_1.pth --train 0
```

---

## 🚀 Training

📌 **Note:** 

if you only use a NVIDIA RTX 3090,please run the following command

```bash
python train.py --split samearea --batch_size 96 --lr 1e-4 --epochs 25 --save_dir ./runs/checkpoints
```

if you have a GPU with a bigger memory,just add the batch_size!

To monitor training progress and visualize the loss and evaluation metrics, run:

```
tensorboard --logdir=/your_path/CVL/runs
```
git使用指令教程

# 1. 设置 Git 全局用户名
# 作用：告诉 Git 你是谁。以后的每次提交都会署上这个名字。
# "--global" 表示这台机器上所有的 Git 仓库默认都用这个名字。
git config --global user.name "YaoweiLiu"

# 2. 设置 Git 全局邮箱
# 作用：设置你的联系方式。GitHub 会根据这个邮箱将代码提交记录关联到你的 GitHub 头像和账号上。
git config --global user.email "lyw5208@mail.ustc.edu.cn"

# 3. 初始化本地仓库
# 作用：在当前文件夹下创建一个隐藏的 `.git` 目录，把这个普通文件夹变成一个 Git 可以管理的仓库。
git init

# 4. 添加所有文件到暂存区
# 作用：告诉 Git，“注意这些文件的变化”。"." 代表当前目录下的所有文件。
# 这是提交代码前的准备步骤（相当于把要寄的信放进信封）。
git add .

# 5. 提交文件到本地仓库
# 作用：将暂存区的文件正式保存到本地的历史记录中。
# "-m" 后面跟的是提交说明（Message），这里写的是“Initial commit”（初次提交）。
# （相当于把信封封口并盖上邮戳，投进本地邮箱）。
git commit -m "Initial commit"

# 6. 关联远程仓库
# 作用：给本地仓库添加一个远程地址，起个别名叫 "origin"。
# 以后不需要每次输入长长的 URL，直接用 "origin" 代指这个 GitHub 仓库。
git remote add origin https://github.com/YWliu040120/CVL.git

# 7. 创建并切换到 main 分支
# 作用：创建一个名为 "main" 的新分支，并立即切换过去。
# GitHub 现在默认的主分支叫 "main"（以前叫 master），为了保持一致，这里手动指定分支名。
git checkout -b main

# 8. 查看仓库状态
# 作用：检查一下现在还有没有没提交的文件，或者确认一下当前在哪个分支上。
# 这是一个好习惯，用来确认一切就绪。
git status

# 9. 推送到远程仓库
# 作用：把本地 "main" 分支的代码上传到远程 "origin" 的 "main" 分支。
# "-u" 的作用是建立“上游”关联，以后你再推代码只需输入 "git push" 即可，不用再输这么长了。
git push -u origin main