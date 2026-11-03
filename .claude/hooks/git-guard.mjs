#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

const TYPES = 'feat|fix|chore|docs|refactor|perf|test|build|ci|style'
const COMMIT_RE = new RegExp(`^(${TYPES})(\\([a-z0-9-]+\\))?!?: [a-z0-9].*[^.\\s]$`)
const BRANCH_RE = /^(feat|fix|chore|docs|refactor|perf|test)\/[a-z0-9][a-z0-9-]*$/
const PROTECTED = new Set(['main', 'master'])
const FORBIDDEN_TEXT = /co-authored-by|generated with|signed-off-by|claude-session/i
const MAX_SUBJECT = 72
const MAX_PR_BODY_LINES = 3

function block(reason) {
  process.stderr.write(`git-guard: ${reason}\n`)
  process.exit(2)
}

function readInput() {
  try {
    return JSON.parse(readFileSync(0, 'utf8'))
  } catch {
    return null
  }
}

function splitCommands(src) {
  const segments = []
  let tokens = []
  let cur = ''
  let has = false
  let i = 0
  const push = () => {
    if (has) tokens.push(cur)
    cur = ''
    has = false
  }
  const end = () => {
    push()
    if (tokens.length) segments.push(tokens)
    tokens = []
  }
  while (i < src.length) {
    const c = src[i]
    if (c === "'") {
      const j = src.indexOf("'", i + 1)
      const stop = j === -1 ? src.length : j
      cur += src.slice(i + 1, stop)
      has = true
      i = stop + 1
      continue
    }
    if (c === '"') {
      i++
      while (i < src.length && src[i] !== '"') {
        if (src[i] === '\\' && i + 1 < src.length && '"\\$`'.includes(src[i + 1])) {
          cur += src[i + 1]
          i += 2
          continue
        }
        cur += src[i]
        i++
      }
      has = true
      i++
      continue
    }
    if (c === '\\' && i + 1 < src.length) {
      if (src[i + 1] !== '\n') cur += src[i + 1]
      has = true
      i += 2
      continue
    }
    if (c === '&' && src[i + 1] === '&') {
      end()
      i += 2
      continue
    }
    if (c === '|' && src[i + 1] === '|') {
      end()
      i += 2
      continue
    }
    if (c === ';' || c === '|' || c === '\n') {
      end()
      i++
      continue
    }
    if (c === ' ' || c === '\t') {
      push()
      i++
      continue
    }
    cur += c
    has = true
    i++
  }
  end()
  return segments
}

function stripPrefix(tokens) {
  let i = 0
  while (i < tokens.length && /^[A-Za-z_][A-Za-z0-9_]*=/.test(tokens[i])) i++
  while (i < tokens.length && ['command', 'exec', 'time', 'env'].includes(tokens[i])) i++
  return tokens.slice(i)
}

function gitSubcommand(tokens) {
  let i = 1
  while (i < tokens.length && tokens[i].startsWith('-')) {
    if (['-C', '-c', '--git-dir', '--work-tree', '--namespace'].includes(tokens[i])) i += 2
    else i++
  }
  return { sub: tokens[i], args: tokens.slice(i + 1) }
}

function currentBranch(cwd) {
  try {
    return execFileSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch {
    return ''
  }
}

function checkSubject(msg, what) {
  if (msg.includes('\n')) block(`${what} must be a single line with no body.`)
  if (FORBIDDEN_TEXT.test(msg)) block(`${what} must not contain attribution or trailer lines.`)
  if (msg.includes('$(') || msg.includes('`')) block(`${what} must be a literal string, not a command substitution.`)
  if (msg.length > MAX_SUBJECT) block(`${what} is ${msg.length} chars; keep it to ${MAX_SUBJECT}.`)
  if (!COMMIT_RE.test(msg))
    block(
      `${what} "${msg}" must look like "type(scope): lowercase subject" with type one of ${TYPES.replaceAll('|', ', ')} and no trailing period.`,
    )
}

function checkBranchName(name) {
  if (!BRANCH_RE.test(name))
    block(`branch "${name}" must be feat/, fix/, chore/, docs/, refactor/, perf/ or test/ followed by lowercase-hyphenated words, e.g. feat/p032-residents-layer.`)
}

function checkCommit(args, cwd) {
  const messages = []
  let noEdit = false
  for (let i = 0; i < args.length; i++) {
    const a = args[i]
    if (a === '--no-verify' || a === '-n') block('never skip hooks with --no-verify.')
    if (a === '-F' || a.startsWith('--file') || /^-[a-zA-Z]*F/.test(a)) block('use a single -m "type(scope): subject" instead of a message file.')
    if (a === '--no-edit') noEdit = true
    if (a === '-m' || a === '--message') {
      messages.push(args[++i] ?? '')
      continue
    }
    if (a.startsWith('--message=')) {
      messages.push(a.slice('--message='.length))
      continue
    }
    const short = /^-([a-zA-Z]*)m(.*)$/.exec(a)
    if (short && !a.startsWith('--')) {
      if (short[2]) messages.push(short[2])
      else messages.push(args[++i] ?? '')
    }
  }
  const branch = currentBranch(cwd)
  if (PROTECTED.has(branch)) block(`do not commit on ${branch}; create a branch first (feat/..., fix/..., chore/...).`)
  if (messages.length === 0 && !noEdit) block('commit with exactly one -m "type(scope): subject"; editors are not available.')
  if (messages.length > 1) block('use exactly one -m; commit messages have no body.')
  if (messages.length === 1) checkSubject(messages[0], 'commit message')
}

function checkPush(args, cwd) {
  for (const a of args) {
    if (a === '--force' || a === '-f' || a.startsWith('--force') || /^-[a-zA-Z]*f/.test(a)) block('force pushes are not allowed.')
    if (a === '--no-verify') block('never skip hooks with --no-verify.')
    if (a.startsWith('+')) block('force pushes are not allowed.')
    const target = a.includes(':') ? a.split(':').pop() : a
    if (PROTECTED.has(target) || PROTECTED.has(target.replace(/^refs\/heads\//, '')))
      block('do not push to main; open a PR and squash-merge it.')
  }
  const positional = args.filter((a) => !a.startsWith('-'))
  if (positional.length < 2 && PROTECTED.has(currentBranch(cwd))) block('do not push from main; work on a branch.')
}

function checkBranchCreation(sub, args) {
  if (sub === 'checkout' || sub === 'switch') {
    const flagIdx = args.findIndex((a) => ['-b', '-B', '-c', '-C', '--create', '--force-create'].includes(a))
    if (flagIdx !== -1 && args[flagIdx + 1]) checkBranchName(args[flagIdx + 1])
  }
  if (sub === 'branch') {
    const positional = args.filter((a) => !a.startsWith('-'))
    const listing = args.some((a) => /^(-d|-D|--delete|-m|-M|--move|-l|--list|-a|-r|--show-current|--contains|--merged|--no-merged|-u|--set-upstream-to|--unset-upstream|-v|-vv)$/.test(a))
    if (!listing && positional.length >= 1) checkBranchName(positional[0])
  }
  if (sub === 'worktree' && args[0] === 'add') {
    const flagIdx = args.findIndex((a) => a === '-b' || a === '-B')
    if (flagIdx !== -1 && args[flagIdx + 1]) checkBranchName(args[flagIdx + 1])
  }
}

function flagValue(args, names) {
  for (let i = 0; i < args.length; i++) {
    for (const n of names) {
      if (args[i] === n) return args[i + 1] ?? ''
      if (args[i].startsWith(`${n}=`)) return args[i].slice(n.length + 1)
    }
  }
  return undefined
}

function checkGh(tokens) {
  const [, group, action, ...args] = tokens
  if (group !== 'pr') return
  if (action === 'create') {
    const title = flagValue(args, ['--title', '-t'])
    const fill = args.some((a) => a === '--fill' || a === '-f' || a === '--fill-first')
    if (title === undefined && !fill) block('gh pr create needs --title "type(scope): subject" and a short --body.')
    if (title !== undefined) checkSubject(title, 'PR title')
    const body = flagValue(args, ['--body', '-b'])
    if (body !== undefined) {
      if (FORBIDDEN_TEXT.test(body)) block('PR body must not contain generated-by or attribution lines.')
      if (body.trim().split('\n').length > MAX_PR_BODY_LINES) block(`PR body is at most ${MAX_PR_BODY_LINES} short lines (what, why, how tested).`)
    }
    if (flagValue(args, ['--body-file', '-F']) !== undefined) block('pass a short --body instead of a body file.')
  }
  if (action === 'merge') {
    if (args.some((a) => a === '--merge' || a === '-m' || a === '--rebase' || a === '-r'))
      block('merge PRs with --squash only.')
    if (!args.some((a) => a === '--squash' || a === '-s')) block('merge PRs with gh pr merge --squash --delete-branch.')
    const subject = flagValue(args, ['--subject', '-t'])
    if (subject !== undefined) checkSubject(subject.replace(/ \(#\d+\)$/, ''), 'squash subject')
  }
}

const input = readInput()
if (!input || input.tool_name !== 'Bash') process.exit(0)
const command = input.tool_input?.command ?? ''
if (!/\b(git|gh)\b/.test(command)) process.exit(0)
const cwd = input.cwd || process.env.CLAUDE_PROJECT_DIR || process.cwd()

if (/\bgit\b[^\n;&|]*\bcommit\b/.test(command) && command.includes('<<'))
  block('commit messages are one line; use -m "type(scope): subject" instead of a heredoc.')

for (const raw of splitCommands(command)) {
  const tokens = stripPrefix(raw)
  if (!tokens.length) continue
  if (tokens[0] === 'git') {
    const { sub, args } = gitSubcommand(tokens)
    if (sub === 'commit') checkCommit(args, cwd)
    else if (sub === 'push') checkPush(args, cwd)
    else checkBranchCreation(sub, args)
  } else if (tokens[0] === 'gh') {
    checkGh(tokens)
  }
}
process.exit(0)
