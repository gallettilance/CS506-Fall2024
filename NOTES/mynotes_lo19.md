Git
A tool to help us manage the timeline(s) of a project (also called repository)
Formally called a version control system or source control management
Fundamental Workflow
As we change the project over time
Create save points (called commits) that track the timeline of the project’s evolution

Creates a timeline of your code/files

Demo
mkdir git-demo
cd git-demo
git init //initializes the git repository
ls -a //shows . .. .git
if you do rm -r .git you are no longer a git repository
can add a README.md file into the folder of git-demo
when doing ls you will get README.md
if you do git status it says the untracked files and stuff
git add README.md or git add . will add everything in the current directory
git commit -m “message”
git log (to look at commit messages, to exit press Q)
git diff (shows you the differences you made in the commit)
git log (shows you the log)
git checkout [with specific id] //goes back in time in the repo
git checkout main //goes back to the current state
GitHub vs Git
Git → [terminal] a version control system
GitHub → [browser] a website to backup and host the timeline(s) of your project
Fundamental Workflow
Create save points (called commits)
Push the updates to GitHub (from your laptop) to back up your work

Initialize a repository
git init
git add <files>
git commit -m “some message”

Adding a remote that points to GitHub
git remote add origin <link>



DEMO
git remote add origin [e.x. git@github.com:gallettilance/git-demo.git
git remote -v //shows the origin fetch and push


Motivation
For each project (repository) I own, I want to write code where:
1. Iterating on (+keeping track of) different versions of the code is easy
2. Work is backed up to and hosted on the cloud
3. Collaboration is productive
Iterating on different versions
The ease or difficulty of adding a new feature to the codebase might depend on the state/version of the codebase
It may be easiest to add this feature at a specific commit



This won’t work
Another way: (will branch off that particular commit to create a new timeline)
Can push commits per branch

Can create a lot of branches
But one branch typically needs to be chosen as the primary, stable branch
This branch is typically called the “main” branch
At some point we will want to clean up certain branches by merging them with the master/main branch or with each other
Merging is trivial if the base of one branch is the head of the other, the changes are “simply” appended
When this is not the case, commits can conflict with each other

We need to change the base of the login-page branch (rebase) to be at the head of the master branch


This is not a simple operation! It will often require manual intervention to resolve the conflicts
