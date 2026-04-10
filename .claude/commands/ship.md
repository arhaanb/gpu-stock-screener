Review all uncommitted changes (staged, unstaged, and untracked files). Create a single commit with a conventional commit message, then push.

Steps:
1. Run `git status` and `git diff` to understand all changes
2. Stage all relevant changed and new files (do NOT stage .env or credential files)
3. Write a commit message following this format:
   - First line: `<type>: <short summary>` (max 72 chars)
   - Blank line
   - Description body: simple lowercase bullet points explaining what changed and why
   - Types: `feat:` (new feature), `fix:` (bug fix), `refactor:` (code restructuring), `style:` (UI/styling), `chore:` (maintenance/config), `docs:` (documentation), `perf:` (performance), `test:` (tests)
   - Use lowercase for the summary, no period at the end
   - The description should provide context that isn't obvious from the diff
4. Commit the changes using a HEREDOC for the message
5. Push to remote: `git push origin main`
