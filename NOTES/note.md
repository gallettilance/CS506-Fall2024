
---
- Git
	- A tool to help us manage the timeline of a project (also called repository)
	- Create save points (commits) that track the timeline of the project's evolution
	- GitHub vs. Git
		- Git -- [terminal] a version control system
		- GitHub -- [browser] a website to backup and host the timeline(s) of your project
	- Push the updates to GitHub to back up your work
- Commands:
	- Initialize a repo: `git init`
	- Add and commit changes: `git add <files>` -> `git commit -m "some message"`
	- Add a remote that points to GitHub: `git remote add origin <link>`

- Iterating on Different Versions
	- We will branch oﬀ of that particular commit to create a new timeline
	- We can push commits per branch
	- We can create lots of branches, but one branch needs to chosen as the primary, stable branch (main)
	- Other branches are usually named after either the feature that is being developed on or the major or minor version of the software / product
		- ![[Pasted image 20240911173239.png]]
	- At some point we will want to clean up certain branches by **merging** them with the master / main branch or with each other.
		- Merging is trivial if the base of one branch is the head of the other - the changes are simply appended.
		- When this is not the case, commits can conflict with each other
			- We need to change the **base** of the login-page branch (**rebase**) to be at the **head** of the master branch

- Collaboration
	- Make a copy (**fork**) of the main repo
	- Make all changes they want to this copy
	- Request that part of their copy be merged into the main repo via a Pull Request (PR)