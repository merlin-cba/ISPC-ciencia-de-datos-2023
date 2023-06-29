import openai

class Chat:
    def __init__(self):
        openai.organization = "org-8e3WI3uJlnUQdVxtlZvF5Ba2"
        openai.api_key = "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"
    def chatgpt(self, input):
        self.completion = openai.Completion.create(engine = "text-davinci-003",
                                                   prompt = input,
                                                   max_tokens = 2048)
        return f'{ self.completion.choices[0].text}'




