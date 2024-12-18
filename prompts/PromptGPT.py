class PromptGPT:
    def __init__(self, granularity_level):
        self.granularity_level = granularity_level

    def get_task_guideline(self):
        if self.granularity_level=="paragraph":
            return """Follow this guideline to write up an explanation before making a final prediction:
Predicting the next paragraph involves a deeper understanding of the broader structure and purpose of the text, including how ideas are expanded, contrasted, or concluded over several sentences. Here’s a step-by-step guide:

1. Understand the Broader Context and Main Theme  
   - Identify Core Ideas: Determine the central topic or message of the text so far. Ask:
     - What is the text as a whole trying to convey?
     - Is it focused on explaining, arguing, telling a story, or describing a concept?
   - Extract Key Concepts and Vocabulary: Focus on the main ideas, technical terms, or important details that could carry forward into the next paragraph.
     - Are there key points or ideas that feel partially explained or ready for further development?

2. Analyze the Structure and Logical Flow of Previous Paragraphs 
   - Detect the Function of Each Paragraph: Consider how each paragraph contributes to the overall argument or narrative:
     - Does the paragraph introduce a topic, provide background, offer evidence, illustrate an example, or summarize?
   - Look for Patterns or Progressions in the Structure: Many texts follow logical or narrative sequences, such as:
     - Chronological Order: If events or steps are being recounted, the next paragraph might naturally progress to the following event.
     - Cause-and-Effect: If the paragraph ends with a cause, the next one might explain the effect.
     - Argumentative Progression: An argument may start with a claim, followed by evidence, then a counterargument or rebuttal. Predict if the next paragraph will reinforce, challenge, or resolve this.
   - Identify Transitional Sentences: Often, paragraphs end with sentences that hint at what’s next (e.g., “This leads us to…” or “As a result…”), signaling the content of the next paragraph.

3. Consider the Tone, Style, and Audience
   - Maintain Tone and Style Consistency: Predict the tone of the next paragraph by noting whether the text is formal, conversational, objective, or persuasive.
   - Match Sentence Complexity and Length: If previous paragraphs used simple sentences and a direct tone, it’s likely the next paragraph will be similar. Alternatively, complex sentence structures might signal a more in-depth or technical discussion.
   - Understand the Audience’s Needs: The complexity and specificity of the next paragraph should align with the target audience. For a scientific article, expect detailed explanations; for a blog, more accessible language.

4. Identify Gaps, Open Questions, and Logical Next Steps  
   - Locate Unanswered Questions or Incomplete Ideas: Ask if the text has raised issues that haven’t been fully addressed. The next paragraph could naturally expand on these.
   - Look for Counterpoints or Rebuttals: If the paragraph strongly argues one side, the next might balance it with an alternative view or counterargument, especially in an objective or balanced discussion.
   - Anticipate Elaboration or Examples: If a complex concept was briefly introduced, the next paragraph might dive deeper, provide examples, or break down the concept in a more understandable way.

5. Use Predictive Language Cues 
   - Look for Transition Words or Phrases that Indicate New Ideas: Certain words or phrases in the existing text might suggest what’s to come:
     - Additive Transitions (e.g., “furthermore,” “in addition”) suggest that the next paragraph will build on the previous one.
     - Contrastive Transitions (e.g., “however,” “on the other hand”) suggest a counterpoint or an alternative perspective.
     - Causal Transitions (e.g., “therefore,” “as a result”) indicate that the next paragraph might explore the consequences or implications of what was just discussed.
   - Refer to Pronouns or Linking Phrases: Words like “this,” “such as,” or “another example” could imply that the next paragraph will continue elaborating on the same topic or give further illustrations.

6. Apply Genre-Specific Patterns
   - Argumentative Texts: In persuasive or argumentative writing, paragraphs often follow a claim-evidence-reasoning pattern. If a claim has been presented, the next paragraph might present evidence or expand on the reasoning.
   - Narrative Texts: In storytelling, paragraphs usually follow a sequence of events, reactions, or emotional shifts. After a moment of action, expect a response or reflection.
   - Expository or Informative Texts: Informational texts may follow a problem-solution or topic-detail-example format. If an issue was presented, the next paragraph might discuss the solution, implications, or give further context.

7. Visualize and Draft a Concept for the Next Paragraph
   - After analyzing the text, try to draft a conceptual idea of what the next paragraph could cover. Think of it as answering the following questions:
     - How does this text logically progress?
     - What concept, example, or counterpoint would feel most natural to introduce?
   - Construct a mental outline for the next paragraph’s content, keeping in mind coherence, flow, and the overall objective.

By following these steps, you’ll produce a paragraph that logically and stylistically continues from the original, making it difficult for readers to distinguish where the author’s work ends and yours begins."""
        else: 
            raise ValueError("Granularity level {} not supported!".format(self.granularity_level))

    def get_task_prompt(self):
        if self.granularity_level=="paragraph":
            return """Given the above piece of text, you are tasked with writing the next paragraph that would logically follow the given text in the same style as the author of the original text such that it can be be passed off as the original text. Think this through before making a prediction. Use the ground truth as reference in crafting your explanation but pretend you did not have access to the ground truth. Make the final prediction after "### Final prediction".
{guideline}""".format(
    guideline=self.get_task_guideline()
)
        else: 
            raise ValueError("Granularity level {} not supported!".format(self.granularity_level))

    def get_user_prompt(self):
        return """### Text
{text}

### Task
{task_prompt}

### Ground Truth
{ground_truth}

### Thoughts"""

    def get_user_message(self, input_text, output_text):  
        return self.get_user_prompt().format(
            text=input_text,
            task_prompt=self.get_task_prompt(),
            ground_truth=output_text
        )