import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

matches = tool.check('can i make a cup green tea for a empty pack ?')
print(matches)