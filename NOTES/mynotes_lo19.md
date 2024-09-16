## 9/11 notes
Git
A tool to help us manage the timeline(s) of a project (also called repository)
Formally called a version control system or source control management
Fundamental Workflow
As we change the project over time
Create save points (called commits) that track the timeline of the project’s evolution

Creates a timeline of your code/files

Demo
-mkdir git-demo
- cd git-demo
- git init //initializes the git repository
- ls -a //shows . .. .git
- if you do rm -r .git you are no longer a git repository
- can add a README.md file into the folder of git-demo
- when doing ls you will get README.md
- if you do git status it says the untracked files and stuff
- git add README.md or git add . will add everything in the current directory
- git commit -m “message”
- git log (to look at commit messages, to exit press Q)
- git diff (shows you the differences you made in the commit)
- git log (shows you the log)
- git checkout [with specific id] //goes back in time in the repo
- git checkout main //goes back to the current state
GitHub vs Git
- Git → [terminal] a version control system
- GitHub → [browser] a website to backup and host the timeline(s) of your project
Fundamental Workflow
- Create save points (called commits)
- Push the updates to GitHub (from your laptop) to back up your work

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


## 9/16 notes
Each of the commits should be somewhat meaningful
The more organized/clear the commits, the easier to look through the history and figure out what was going on
Can always git clone a repo to get a fresh copy of your work

When working with multiple people on a repo:
Create a copy/branch, work on whatever you need to, then submit a pull request onto the main
After a vetting process, you either get to merge your code or you don't
Repository as branches

GitHub main repository -(fork)-> GitHub my repository -(git clone)-> my repository on laptop (add+commit)
If new changes are made to the main repository you won't get access to them unless you specifically specify that you want to get those changes
Remote "ORIGIN" is my repository that was forked off
Remote "UPSTREAM" is the main repository on GitHub
Git push ORIGIN -> pushes to my repo

In GitHub you need to visualize the timelines
Different repos as different branches part of the same whole (like with the above example)
There is upstream/main on the main repo, origin/main on the github site my repo, and main on my repo on the laptop
If the upstream/main updates after you have forked, you can do the following to update your copy:
- Git pull upstream main
- Git push origin main (pushing to your personal repo on the Github website)

Best practices:
Never really commit to the main branch (its supposed to be special and stable, not where you do development) -> want to do development in an isolated area

Good first issue tag are good first issue things on repos (like on the tensorflow repo)
(END OF GIT SLIDES)

Data science is hard (likely an impossible task most of the time)
Sometimes not clear if you've found all the factors that influence something
Sometimes impossible to quantify/capture everything; might be inherent randomness to things
Very rare to find things that are perfectly related to each other

Example of linear relationship between x and y (y=f(x)) -> Pressure and temperature
- if there is all of a sudden one singular outlier, your hypothesis y=f(x) (perfectly linear relationship) is now completely wrong

Can create models that are somewhat useful
- EX: y=f(x) could give us a general guide on how x and y vary

Confirmation Bias
EX: think of a rule that governs triples, and then you have to guess what the rule is (given yes-or-no answers to questions you ask).
- announce "(2,4,6) follows the rule"
- examples submitted by participants: (2,4,3) -> NO; (6,8,10) -> YES; (1,3,5) -> YES
- after submitting these examples, they wrote their hypothesized rule. would you have wanted to try more examples? if so, which and for what reason?
- poll: A. (100,102,104) B. (5,7,9) C. (1,2,3)
challenges of data science
- a set of exdamples may not always be representative of the underlying rule
- there may be infinitely many rules that match the examples provided
- rules and/or examples may change over time

data science is about doing the best with the data that you have a lot of the time, and sometimes the data will be biased in one way
data science is very difficult! all models are wrong but some are useful

positive examples vs negative examples
- assuming the hypothesis h is (x, x+2, x+4) which type of examples are the following?
- (2,4,3) -> negative
- (6,8,10) -> positive
- (1,3,5) -> positive

if you just try positive examples then you will only get positive responses and won't be aware of possible constraints

data science workflow (simplified)
process data -> explore data -(either loop back or continue) -> extract features (loop back or continue) -> create model (loop back)

use common sense first, think about if there's actually an relationship between the things i'm looking at
putting the data in the right format and quantifying it properly is important
modeling part is a very small part of data science --> get the biggest bang for your buck in the first couple of steps (process data and explore data and extract features)
can visualize the relationships between features (if i change the predictor does it also change the outcome? --> if you change the predictor and there's no change that means there is no relationship)

types of data - records
- m-dimensional points/vectors; EX: (name, age, balance) -> ("John", 20, 100)
types of data - graphs
- nodes connected by edges; EX: triangle with nodes and edges that we can represent as an adjacency matrix (says is there an edge btwn the two nodes) or an adjacency list
types of data - images
- image that can be split up into a collection of pixels (usually a triple of red, green, blue intensities)
types of data - text
- list of words that we can extract more info from; like with a corpus of documents and figure out what words are important and which are not