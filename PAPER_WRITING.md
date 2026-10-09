# Scientific Paper Writing Guide

Use this guide to draft or revise technical research papers for clarity, coherence, and scientific precision. The goal is not to make every paper sound the same. It is to make the reasoning easy to follow while preserving the authors' voice and the subject's necessary technical detail.

Treat revision as a careful human line edit, not as a general rewrite. Simplify the explanation itself rather than mechanically replacing technical words with supposedly simpler synonyms.

## 1. Start with the scientific story

Before editing sentences, write the paper's story in five to seven plain statements:

1. What broad problem matters?
2. What is already known?
3. What remains unresolved?
4. What does this paper do?
5. What evidence answers the unresolved question?
6. What is the main result?
7. What remains limited or unknown?

Every major section should advance this story. Remove, shorten, or relocate material that does not help the reader understand or evaluate it.

The storyline is not a list of everything done during the project. It is the shortest defensible chain from the motivating problem to the conclusion supported by the evidence.

### Use a V-shaped structure

The paper should narrow and then widen:

1. Begin with the general scientific problem and why it matters.
2. Narrow to the specific gap, construction, and tests addressed by the paper.
3. End by returning to the broader meaning, limitations, and practical consequences.

Use the same shape within major sections whenever possible. Open with the high-level question and its role in the paper, move into the necessary technical detail, and close with the answer and a transition to the next question. This is a logical structure, not a demand for artificial symmetry.

### Respect the reader's knowledge order

Present ideas in the order needed to understand them. Do not refer to a proof, proposition, mechanism, result, or technical distinction before it has been introduced. The abstract and introduction may preview the main outcome in ordinary language, but they should not depend on later notation or unexplained labels.

A transition should normally identify the question that remains after the current section. It should motivate the next section without giving its detailed answer in advance. At every transition, ask what the reader knows at that point and what they need to learn next.

## 2. Separate prior work from the present contribution

Make the boundary unmistakable.

- Describe established knowledge in neutral, factual prose.
- Explain the specific gap before presenting the new work.
- Begin the contribution with a clear transition such as “Here, we…” or “In this work, we…”.
- Use active voice for the authors' choices, derivations, tests, and conclusions.
- Make the change visible in both structure and voice. Prior work may be described mostly in neutral or passive language where natural. The present contribution should switch clearly to active language using “we.”
- Do not hide this change in the middle of a paragraph. Start a new paragraph, subsection, or section when the scale of the paper permits it.
- Do not imply novelty merely by describing standard material in new terminology.

A useful introduction progression is:

1. Motivate the general problem.
2. Explain the established approaches relevant to that problem.
3. Identify the limitation or unresolved question.
4. State what this paper contributes.
5. Preview the principal result and its scope.

Do not turn the contribution paragraph into a checklist of sentences beginning with “We present,” “We derive,” or “We demonstrate.” Group related contributions into a short argument.

### Cite ideas rather than narrating authors

Discuss the scientific concept or result and place the citation directly after it. Avoid humanities-style narration built around researchers' names.

Prefer:

> Rank-adaptive tensor-network integrators enlarge the represented space before truncation [7, 8].

Avoid:

> Smith and Jones introduced a rank-adaptive tensor-network integrator [7].

Use author names only when the identity itself is relevant, such as a named theorem, a direct historical dispute, or wording that cannot otherwise be attributed clearly. A related-work section should compare assumptions, constructions, and results rather than recounting who did what.

## 3. Match every claim to evidence

For each central claim, ask:

- What result supports it?
- Does that result establish the claim directly, or merely illustrate it?
- What alternative explanation has been ruled out?
- What qualification must remain?

Use the weakest claim that communicates the actual result completely.

Examples of important distinctions:

- Numerical tests can validate an implementation or illustrate an identity. They do not prove a universal theorem.
- Decreasing error over a short timestep sequence shows empirical refinement. It does not establish a formal convergence order.
- One controlled example can identify a concrete failure mode. It does not show that the failure occurs for every model or implementation.
- Equal tolerances do not necessarily imply equal cost, equal discarded weight, or an equal-resource comparison.
- A method working after a correction does not imply that it outperforms established alternatives.
- Failure to observe an advantage is a result, not a reason to manufacture a stronger performance story.

Preserve limitations wherever they affect interpretation. Simpler prose must not become stronger prose.

## 4. Write for an informed non-specialist

Assume the reader understands the general field but not this paper's particular construction.

- Explain the problem before naming detailed mathematical objects.
- Introduce notation only when it becomes useful.
- Define each specialized term before relying on it.
- Prefer language already used in the closest literature.
- Keep established terminology when it is both precise and readable. Do not rename a known object merely to make the paper appear novel.
- Use an ordinary description instead of inventing a term for every implementation choice.
- Reserve dense terminology for sections where mathematical precision requires it.
- When two formulations are equally precise, choose the one that is more accessible.

For example, first write “project the coefficients into the updated basis.” Introduce a shorter formal name only if the operation appears often enough that the name genuinely helps.

A term should earn its place. Name a concept when the name improves later reasoning, not simply because the concept exists.

## 5. Use simple, exact language

### Sentence construction

- Put one main idea in each sentence.
- Prefer concrete verbs such as “keep,” “remove,” “compare,” “project,” “increase,” and “compress.”
- Break up sentences containing several conditions, contrasts, or qualifications.
- Avoid noun stacks with several technical modifiers.
- Keep the subject and verb close together.
- State the result before discussing secondary details.
- Use “we” naturally, but not at the beginning of every sentence.

### Words and structures to use sparingly

- Flowery adjectives and promotional modifiers
- “Crucial,” “groundbreaking,” “robust,” “comprehensive,” and “systematic” unless they have a precise meaning
- “Highlights,” “underscores,” “reveals,” “offers insight into,” and “plays a pivotal role” when the result can be stated directly
- Repeated “not merely X, but Y” constructions
- Em dashes
- Heavy use of colons and semicolons
- Formal synonyms where an ordinary word is clearer
- Repeated three-part lists used only for rhetorical rhythm

Replace claims about importance with the reason the result matters. Replace claims that a result “demonstrates” something with the observation and its supported interpretation.

Avoid both extremes: prose should not be ornate, but it should not read like a sequence of clipped notes. Transitions should show how one question leads to the next.

## 6. Build sections around reader questions

At the beginning of each major section, state in one or two sentences why it is needed. Then move from that overview into the technical detail. At the end, return to the section's main answer and connect it to the next question without introducing unexplained material.

Transitions must follow the reader's current knowledge. Do not write “as proved below,” interpret a result that has not yet been shown, or use terminology belonging to the next section. If later material must be previewed, describe only the motivating question in language already available to the reader.

For each theoretical subsection, make clear:

1. What problem is being formalized?
2. What assumptions are required?
3. What is proved?
4. Why is the result needed later?
5. What does the result not establish?

For each numerical study, use this order:

1. The question being tested
2. The construction of the test
3. The observed result
4. The conclusion supported by the observation
5. The conclusion that cannot be drawn

Do not begin with a page of parameters before stating why the calculation exists. Give enough setup for reproducibility, then keep the main observation visible.

## 7. Design the abstract as a compact argument

The abstract should be understandable without the Methods section. A reliable structure is:

1. The broad problem
2. The unresolved issue
3. The paper's approach
4. The central theoretical or methodological contribution
5. The decisive evidence
6. The main conclusion, including an important negative result if relevant

Avoid:

- notation;
- proposition or equation numbers;
- new terminology that is unnecessary to understand the result;
- a list of every experiment;
- claims of novelty or importance unsupported by the abstract itself;
- detailed implementation labels when a plain description works.

The abstract should explain what happens and why it matters, not reproduce the manuscript's internal vocabulary.

## 8. Keep the introduction progressive

The introduction should follow the paper's V shape. It begins with the broad problem, narrows to the exact unresolved question, and ends by placing the contribution and main finding back in the wider context.

It should not spoil results before the reader understands the problem, but it should still state the paper's main finding plainly. Preview the conclusion at a level appropriate for the introduction. Do not refer to a later proof, proposition number, figure, or technical mechanism before establishing the concepts needed to understand it.

Each paragraph should perform one role:

- establish the setting;
- narrow to the relevant methods;
- explain the unresolved problem;
- identify the contribution;
- summarize the evidence and conclusion.

Do not introduce notation, algorithm labels, or fine distinctions before the reader needs them. Cite closely related work where the comparison becomes relevant, and state exactly how the present contribution differs. Frame these citations around the methods or findings rather than the researchers' names.

## 9. Present methods in dependency order

Definitions and operations should appear before anything that depends on them.

- Begin with the minimum common notation.
- Explain each representation before manipulating it.
- Introduce the algorithm in the same order in which it operates.
- Separate exact mathematical statements from implementation conventions.
- Distinguish what holds before approximation or compression from what may fail afterward.
- State whether a choice is mathematically required, one valid implementation, or merely the convention used in the paper.

If a passage cannot be simplified without losing precision, keep the formal language and add one plain explanatory sentence around it.

## 10. Make results answer the paper's claims

The Results section should be an evidence chain, not a log of completed computations.

A useful progression for a methods paper is:

1. Structural or unit-level checks of the derived properties
2. Independent correctness validation of the complete implementation
3. A controlled test isolating the proposed mechanism
4. Ablations ruling out plausible alternatives
5. A restrained comparison with established methods

Report the computational environment and software used. Provide enough information to reproduce the study, including the relevant code revision, parameters, reference construction, and accuracy metrics.

Use negative results directly. If a method is less accurate, slower, or less stable in the tested regime, state that result and adjust the paper's claim. Do not compensate with vague claims of potential superiority.

## 11. Make figures and tables carry arguments

Every figure or table should answer a question that matters to the storyline.

Before adding one, ask:

- What question does this item answer?
- Is a plot, table, or sentence the clearest format?
- What should the reader conclude from it?
- Is that conclusion stated in the surrounding text?

Caption structure:

1. A short bold title
2. What question is addressed or what is shown
3. The essential setup needed to interpret it
4. The main answer
5. Any qualification necessary to prevent overinterpretation

Captions should be understandable on their own but should not reproduce an entire Results paragraph. Introduce every figure and table before interpreting it. Use the same terminology in the caption, legend, table entries, and main text.

Use plots for trends, tradeoffs, or comparisons across several values. Use tables for exact values, configurations, or stage-by-stage states. Do not convert a table into a plot merely to make the paper look more visual.

## 12. Use the discussion for interpretation

Do not repeat the paper section by section. Answer:

1. What was learned?
2. Why does it matter?
3. Which conclusions are limited to this setup?
4. What remains unresolved?

Separate limitations caused by the implementation from limitations of the underlying method. A plausible but unsuccessful algorithmic choice is not automatically a coding error.

End with a restrained statement of practical or scientific relevance. Do not advertise a general advance when the evidence establishes a focused one.

## 13. Preserve the authors' voice

Use prior papers by the same authors to calibrate paragraph length, level of explanation, and preferred transitions. Do not copy their wording or force the current paper into an unrelated template.

Avoid repeated global rewrites. They tend to flatten the voice, introduce new terminology, and produce polished but generic prose. Once the structure is sound, prefer local edits made while reading the paper in order.

Good scientific prose should sound like a knowledgeable author explaining a result carefully, not like a press release and not like an instruction manual.

## 14. Editing workflow

### Pass 1: Story and claims

- Write the paper's story in plain language.
- List the central claims and the evidence supporting each one.
- Remove unsupported claims and disconnected results.
- Check novelty against the closest literature.

### Pass 2: Structure

- Reorder sections and paragraphs by conceptual dependency.
- Check the paper's V shape from broad motivation to specific evidence and back to broader interpretation.
- Apply the same overview-detail-transition pattern within major sections where it helps.
- Make the prior-work/new-work boundary explicit.
- Ensure that the shift to the present work is visible in both section structure and active voice.
- Ensure every section answers a necessary question.
- Remove forward references that require knowledge the reader does not yet have.
- Rewrite transitions so that each section motivates the next question without prematurely answering it.
- Move secondary derivations and diagnostics to appendices when appropriate.

### Pass 3: Language

- Remove filler, hype, and unnecessary adjectives.
- Simplify terminology and long sentences.
- Replace vague verbs with concrete statements.
- Improve transitions without adding rhetorical padding.

### Pass 4: Evidence presentation

- Check every figure, table, and caption against its reader question.
- Confirm that numerical comparisons use appropriate metrics and controls.
- Make negative results and limitations explicit.

### Pass 5: Human read-through

Read the manuscript from beginning to end without editing individual sentences on the first pass. Mark every point where the reader must stop, infer a missing connection, or remember an undefined term. Then repair those points locally.

## 15. Non-negotiable editing constraints

Unless an actual inconsistency is found, do not change:

- equations;
- numerical values;
- citations;
- propositions or mathematical conditions;
- algorithm definitions;
- figure data;
- conclusions required by the evidence.

Flag suspected inconsistencies instead of silently correcting them. Record any scientific changes separately from language edits.

## 16. Final audit

Read the paper in order as if unfamiliar with the work. Confirm that:

- the complete story can be summarized in a short paragraph;
- every section advances that story;
- the paper narrows from general motivation to its specific contribution and widens again to interpretation;
- major sections open with an overview, provide the required detail, and close with an answer or bridge;
- every specialized term is explained before use;
- no paragraph depends on a later definition or result;
- no transition spoils a proof, result, or mechanism that the reader has not encountered;
- the boundary between known work and new work is obvious;
- the present contribution is marked by a clear structural and active-voice shift;
- literature is discussed through concepts and findings rather than author-name narration;
- each major claim has visible supporting evidence;
- limitations are stated where they affect interpretation;
- the abstract is understandable without the Methods;
- terminology is consistent across prose, equations, figures, and tables;
- captions state what is shown and what the reader should conclude;
- negative results have not been hidden or rhetorically softened;
- equations, references, numerical values, and cross-references remain intact;
- code and data availability statements describe what is actually accessible;
- the source compiles without new warnings or layout problems.

## Recommended instruction for a writing assistant

> Edit this manuscript for a coherent scientific storyline, precise claims, and natural technical prose. Write for a reader who understands the broader field but not this paper's particular construction. Use the simplest language that preserves the mathematics. First identify the paper's problem, gap, contribution, evidence, main conclusion, and limitations. Give the paper a V-shaped structure that moves from the broad problem to the specific contribution and evidence, then returns to the broader meaning and limitations. Use the same overview-detail-transition pattern within major sections where appropriate. Ensure that every section advances the storyline in the order the reader needs. Do not mention proofs, results, distinctions, or terminology before they have been introduced. Transitions should motivate the next question without spoiling its answer. Clearly separate established work from the present contribution through a noticeable structural change and a shift from mostly neutral or passive prose to active “we” language. Discuss previous literature through concepts and findings followed by citations, not through narration centered on author names. Define specialized terms before use, prefer established language from the closest literature, and avoid inventing labels that do not help later reasoning. Remove filler, promotional adjectives, em dashes, excessive colons or semicolons, repetitive rhetorical structures, and long jargon-heavy sentences. Preserve equations, numerical values, citations, propositions, and mathematical conditions unless an inconsistency is found; flag such inconsistencies rather than silently changing them. Match every claim to its evidence and retain all necessary qualifications. Organize each numerical study around its question, setup, observation, supported conclusion, and limitation. Make every figure and table answer a clear question, with a concise caption that states the main result. Finish with a complete read-through for logical dependencies, terminology, claim scope, compilation, and layout. Return the revised source with a short change log of substantive structural or scientific edits, not a catalogue of wording substitutions.
