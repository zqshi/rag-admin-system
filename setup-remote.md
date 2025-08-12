# 🚀 推送到GitHub远程仓库

## 选择一种方式：

### 方式1：使用HTTPS（推荐新手）

```bash
# 示例URL格式：
git remote add origin https://github.com/你的用户名/rag-admin-system.git

# 然后推送
git push -u origin main
```

### 方式2：使用SSH（推荐）

```bash
# 示例URL格式：
git remote add origin git@github.com:你的用户名/rag-admin-system.git

# 然后推送
git push -u origin main
```

## 常见仓库URL示例：

- GitHub: `https://github.com/username/repo-name.git`
- GitLab: `https://gitlab.com/username/repo-name.git`
- Gitee: `https://gitee.com/username/repo-name.git`
- Coding: `https://e.coding.net/team/project/repo-name.git`

## 如果遇到认证问题：

### GitHub Personal Access Token (PAT)
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" -> "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成token后，使用token作为密码

### 配置credential helper（保存密码）
```bash
git config --global credential.helper store
```