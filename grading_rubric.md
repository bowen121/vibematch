Final project rubric
Project Checkpoint — 10%
Written Proposal — 4%
A single document of two paragraphs, one describing the problem and the other describing methods.
4 pts: Both paragraphs are present and substantive. The problem paragraph clearly identifies a dataset, a user-facing question, and why it's interesting. The methods paragraph names specific algorithms from the course syllabus and explains why they're appropriate for the problem.
2 pts: Both paragraphs present but one is vague or generic.
1 pt: Only one paragraph, or both are a few sentences with no specifics.
0 pts: Missing.
Design Document — 6%
A one-page document covering three things:
Repo structure (2 pts): A directory tree with a one-line description of each component. Full credit if structure is logical and separation of concerns is clear.
Division of labor (2 pts): Each team member is assigned at least one concrete module or deliverable.
Stub code in a public GitHub repo (2 pts): The repo must exist and contain a README, a requirements file, a working environment, and empty placeholder files for each module described in the repo structure.
Final Project — 40%
Working Demo — 10%
Evaluated live during the final presentation session. Graders should attempt to run the app independently before the session using the submitted repo.
10 pts: App runs end-to-end with no live modifications.
8 pts: App runs but requires undocumented steps or minor environment fixes.
5 pts: App partially runs. Some features work, others crash or return errors.
2 pt: App does not run but a recorded demo is provided.
0 pts: Nothing works and no demo is provided.
Live Presentation — 10%
Conceptual explanation (5 pts): Each team member should be able to answer basic questions about the part of the codebase they own.
Presentation clarity (5 pts): Does the team explain the problem, approach, and results in a way that is followable to a classmate?
Algorithm Implementation — 10%
At least one algorithm from the course must be implemented without only being a wrapper around a library. For example, if implementing a mixture of Gaussians, you can’t use the scikit-learn implementation, but you can use numpy and torch.
10 pts: Algorithm is correctly implemented, produces reasonable outputs on the chosen dataset, and the team can explain implementation choices (e.g., choice of distance metric, number of clusters, regularization). The code matches what's being demoed.
8 pts: Implementation is correct but the team struggles to explain one design decision, or there is a minor bug that doesn't affect the demo materially.
4 pts: Algorithm is mostly a library call to scikit-learn or equivalent with a thin wrapper; the team cannot explain internals, or outputs are clearly wrong.
0 pts: The algorithm is entirely copy-pasted from an external source or repo without understanding.
Application Quality — 10%
Data and task meaningfulness (3 pts): Is the dataset real? Is the user-facing question non-trivial? Full credit if the app answers a question a real user would plausibly ask.
UI/UX clarity (3 pts): Can someone unfamiliar with the project use the app without explanation? Full credit if inputs and outputs are labeled, errors are handled gracefully, and the app doesn't require a guided tour. Deduct 1 point for confusing layout or crashes on bad input.
Technical correctness (4 pts): Does the project use course concepts in a coherent way? 4 pts for a correct implementation.
Code Quality — Deductions, up to -10%
Applied after the above scores are totaled.
Unintelligible code (-3%): No comments on non-obvious logic, single-letter variable names, no docstrings.
Duplicated logic (-2%): The same block of code appears in multiple places where a function or class would clearly serve.
Project does not run (-2%)
README is missing (-2%)
Grader’s discretion (-1%)
