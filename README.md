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


> 作用：告诉 Git 你是谁。这台机器上所有的仓库默认都会使用这个身份。

* **1.设置用户名**
    ```bash
    git config --global user.name "YaoweiLiu"
    ```

* **2.设置邮箱**
    ```bash
    git config --global user.email "your_email@example.com"
    ```

* **3.初始化本地仓库**
    ```bash 
    在当前文件夹下创建一个隐藏的 `.git` 目录，把这个普通文件夹变成一个 Git 可以管理的仓库
    git init
    ```

* **4. 添加所有文件到暂存区**
    ```bash 
    告诉 Git，“注意这些文件的变化”。"." 代表当前目录下的所有文件。
    git add .
    ```

* **5. 提交文件到本地仓库**
    ```bash 
    将暂存区的文件正式保存到本地的历史记录中。
    git commit -m "Initial commit"
    ```

* **6. 关联远程仓库**
    ```bash 
    给本地仓库添加一个远程地址，起个别名叫 "origin"。
    git remote add origin https://github.com/YWliu040120/CVL.git
    ```

* **7. 创建并切换到 main 分支**
    ```bash 
    创建一个名为 "main" 的新分支，并立即切换过去。
    git checkout -b main
    ```

* **8. 查看仓库状态**
    ```bash 
    检查一下现在还有没有没提交的文件，或者确认一下当前在哪个分支上。
    git status
    ```

* **9. 推送到远程仓库**
    ```bash 
    把本地 "main" 分支的代码上传到远程 "origin" 的 "main" 分支。
    git push -u origin main
    ```

* **10. 修改后的提交**
    ```bash 
    先git status看一下哪些文件进行了修改
    再git add 你要提交的文件
    再git commit -m "your description for your modify"
    接着git push -u origin main即可
    ```



