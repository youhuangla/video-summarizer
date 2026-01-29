#!/usr/bin/env python3
"""一键创建 GitHub 仓库并推送代码"""
import os
import sys
import subprocess
import json
import urllib.request
import urllib.error

def run(cmd, cwd=None):
    """运行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding='utf-8')
    return result.returncode == 0, result.stdout, result.stderr

def create_github_repo(token, repo_name, description="", private=False):
    """使用 GitHub API 创建仓库"""
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            return True, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return False, json.loads(e.read().decode('utf-8'))

def main():
    print("=" * 60)
    print("GitHub 仓库创建与推送工具")
    print("=" * 60)
    
    # 获取 GitHub Token
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        token = input("\n请输入 GitHub Personal Access Token: ").strip()
    
    if not token:
        print("错误: 需要提供 GitHub Token")
        print("请访问 https://github.com/settings/tokens 创建（勾选 repo 权限）")
        sys.exit(1)
    
    # 获取用户名
    username = input("请输入 GitHub 用户名: ").strip()
    if not username:
        print("错误: 需要提供 GitHub 用户名")
        sys.exit(1)
    
    # 仓库配置
    repo_name = input("仓库名称 [video-summarizer]: ").strip() or "video-summarizer"
    description = input("仓库描述 [使用本地 VLM 生成带时间戳的视频摘要工具]: ").strip() or "使用本地 VLM 生成带时间戳的视频摘要工具"
    is_private = input("是否私有仓库? [y/N]: ").strip().lower() == 'y'
    
    print(f"\n[1/4] 正在创建 GitHub 仓库 '{repo_name}'...")
    success, result = create_github_repo(token, repo_name, description, is_private)
    
    if not success:
        print(f"创建失败: {result.get('message', '未知错误')}")
        if 'errors' in result:
            for err in result['errors']:
                print(f"  - {err.get('message', err)}")
        sys.exit(1)
    
    repo_url = result['html_url']
    git_url = result['clone_url']
    print(f"✓ 仓库创建成功: {repo_url}")
    
    # 项目目录
    project_dir = r"C:\diy_tools\github\video-summarizer"
    
    print(f"\n[2/4] 配置本地 Git 仓库...")
    # 设置本地 git 配置
    run(f'git config --local user.name "{username}"', cwd=project_dir)
    run(f'git config --local user.email "{username}@users.noreply.github.com"', cwd=project_dir)
    
    # 添加远程仓库
    remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    run("git remote remove origin 2>nul", cwd=project_dir)
    success, out, err = run(f"git remote add origin {remote_url}", cwd=project_dir)
    if not success:
        print(f"添加远程仓库失败: {err}")
        sys.exit(1)
    print("✓ 远程仓库配置完成")
    
    print(f"\n[3/4] 提交代码...")
    # 添加所有文件
    run("git add -A", cwd=project_dir)
    
    # 提交
    success, out, err = run('git commit -m "Initial commit: video summarizer project"', cwd=project_dir)
    if not success and "nothing to commit" not in out.lower():
        print(f"提交失败: {err}")
        # 继续尝试推送，可能已经提交过
    
    print("✓ 代码已提交")
    
    print(f"\n[4/4] 推送到 GitHub...")
    # 设置 main 分支
    run("git branch -M main", cwd=project_dir)
    
    # 推送
    success, out, err = run("git push -u origin main", cwd=project_dir)
    if not success:
        print(f"推送失败: {err}")
        sys.exit(1)
    
    print("✓ 推送成功！")
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"仓库地址: {repo_url}")
    print(f"克隆命令: git clone {git_url}")
    print("=" * 60)

if __name__ == "__main__":
    main()
