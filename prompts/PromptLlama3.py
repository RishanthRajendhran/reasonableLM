class PromptLlama3:
    def __init__(self):
        self.message_format = {
            "direct": {
                "header": """""",
                "body": """{content}""",
                "body_contd": """{content}"""
            },
            "explain": {
                "header": """<|start_header_id|>{role}<|end_header_id|>
""",
                "body": """{content}<|eot_id|>""",
                "body_contd": """{content}""",
            },
        } 

        self.task_prompt = {
            "direct": {
                "paragraph": """"""
            },
            "explain": {
                "paragraph": """Given the above piece of text, you are tasked with writing the next paragraph that would logically follow the given text in the same style as the author of the original text such that it can be be passed off as the original text. Think this through before making a prediction."""
            }
        }
    
    def get_task_prompt(self, prompt_type, granularity_level):
        if prompt_type=="direct":
            return self.task_prompt["direct"][granularity_level]
        elif "explain" in prompt_type:
            if "rationalize" in prompt_type:
                return self.task_prompt["explain"][granularity_level] + """Use the ground truth as reference in crafting your thoughts but pretend you did not have access to the ground truth. Make the final prediction after "### Final prediction"."""
            else:
                return self.task_prompt["explain"][granularity_level] + """ Make the final prediction after "### Final prediction"."""
        else: 
            raise ValueError("Unrecognized prompt_type {}".format(prompt_type))

    def get_prompt_template(self, role, instance, model_type, prompt_type, granularity_level):
        if model_type=="base":
            if role=="user":
                return """{text}""".format(**instance)
            elif role=="assistant":
                return """{next_token}""".format(**instance)
            else: 
                raise ValueError("Unrecognized role {}".format(role))
        elif model_type=="instruct":
            if prompt_type=="direct":
                if role=="user":
                    return """{text}""".format(**instance)
                elif role=="assistant":
                    return """{next_token}""".format(**instance)
                else: 
                    raise ValueError("Unrecognized role {}".format(role))
            elif "explain" in prompt_type:
                if role=="user":
                    explain_prompt = """### Text
{text}

### Task
{task_prompt}
""".format(
    **instance,
    task_prompt=self.get_task_prompt(prompt_type, granularity_level)
)    
                    if prompt_type.endswith("_rationalize"):
                        explain_prompt += """
### Ground Truth
{next_token}""".format(**instance)
                    return explain_prompt
                elif role=="assistant":
                    if prompt_type.startswith("explain"):
                        return """### Thoughts
{explanation}

### Final prediction
{next_token}""".format(**instance)
                    elif prompt_type.startswith("generate_"):
                        if "_explain" in prompt_type:
                            return """### Thoughts
"""
                        elif prompt_type.endswith("_answer"):
                            return """### Thoughts
{explanation}

### Final prediction
""".format(**instance)
                        else: 
                            raise ValueError("Unrecognized prompt_type {}".format(prompt_type))
                    else: 
                        raise ValueError("Unrecognized prompt_type {}".format(prompt_type))
                else: 
                    raise ValueError("Unrecognized role {}".format(role))
            else: 
                raise ValueError("Unrecognized prompt_type {}".format(prompt_type))
        else: 
            raise ValueError("Unrecognized model_type {}".format(model_type))

    def get_message(self, prompt_type, role, content, omit_body=False):
        prompt_type = "explain" if "explain" in prompt_type else "direct"
        message_header = self.message_format[prompt_type]["header"].format(role=role)
        if prompt_type.startswith("generate_"):
            message_body = self.message_format[prompt_type]["body_contd"].format(content=content) 
        else: 
            message_body = self.message_format[prompt_type]["body"].format(content=content)
        if omit_body:
            return message_header
        else:
            return message_header + message_body

    def get_prompt(self, instance, is_instruct, granularity_level, prompt_type="explain", return_messages=False, omit_last_body=False, rationalize=False):
        assert prompt_type in [
            "direct", 
            "explain", 
            "explain_rationalize", 
            "generate_explain", 
            "generate_explain_rationalize", 
            "generate_answer", 
            "explain_answer",
        ], "Unrecognized prompt_type: {}".format(prompt_type)
        if rationalize:
            assert prompt_type in ["explain", "generate_explain"]
            prompt_type += "_rationalize"
        messages = []
        if is_instruct:
            messages.append(
                self.get_message( 
                    prompt_type=prompt_type,
                    role="user",
                    content=self.get_prompt_template(
                        role="user",
                        instance=instance,
                        model_type="instruct",
                        prompt_type=prompt_type,
                        granularity_level=granularity_level
                    )
                )
            )

            messages.append(
                self.get_message(
                    prompt_type=prompt_type,
                    role="assistant",
                    content=self.get_prompt_template(
                        role="assistant",
                        instance=instance,
                        model_type="instruct",
                        prompt_type=prompt_type,
                        granularity_level=granularity_level
                    ),
                    omit_body=omit_last_body
                )
            )   
        else:
            assert not rationalize, "rationalize is not meaningful for base completion"
            assert not omit_last_body, "omit_last_body is not meaningful for base model"
            assert not return_messages, "Message format not supported for base model" 
            assert prompt_type=="direct", "Explanations not supported for base models"
            messages.append(
                self.get_prompt_template(
                    role="user",
                    instance=instance,
                    model_type="base",
                    prompt_type=prompt_type
                )
            )
            messages.append(
                self.get_prompt_template(
                    role="user",
                    instance=instance,
                    model_type="base",
                    prompt_type=prompt_type
                )
            )
        if return_messages:
            return messages
        return "".join(messages)