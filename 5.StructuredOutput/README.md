# Structured Output - Basic Examples

This folder contains **basic** examples of structured output formats using LangChain. These are fundamental approaches to get data from LLMs in specific formats.

## What is Structured Output?

Structured output means getting data from an LLM in a specific, predictable format (JSON, CSV, lists, etc.) instead of just plain text. This makes it easier to parse and use the data programmatically.

## Examples Included

### 1. **Basic String Output** (`1_basic_string_output.py`)
- Simplest form - just plain text response
- No parsing needed
- Good for getting simple answers

### 2. **JSON Structured Output** (`2_json_structured_output.py`)
- Ask LLM to return data as JSON
- Parse using Python's `json` module
- Easy to access nested data
- Best for complex data structures

### 3. **CSV Structured Output** (`3_csv_structured_output.py`)
- Comma-separated values format
- Great for tabular data
- Simple to parse with `split()`
- Easy to convert to pandas DataFrame

### 4. **List Structured Output** (`4_list_structured_output.py`)
- Get data as a numbered or bulleted list
- Remove numbering and extract items
- Good for getting multiple items

### 5. **Key-Value Pairs** (`5_keyvalue_structured_output.py`)
- Simple `key: value` format
- Easy to parse line by line
- Good for simple properties
- Human-readable

### 6. **YAML Format** (`6_yaml_structured_output.py`)
- YAML is human and machine readable
- Supports nested data
- Simple hierarchy representation
- Good for configuration-like data

## How to Use

Each file is standalone. Just run:

```bash
python 1_basic_string_output.py
python 2_json_structured_output.py
python 3_csv_structured_output.py
# ... and so on
```

## Key Concepts

✅ **Prompt Engineering**: The prompt tells the LLM exactly what format to use
✅ **Output Parsing**: Python code extracts data from the formatted response
✅ **Error Handling**: Always validate that parsing was successful
✅ **Fallback**: Have logic to handle parsing failures

## Best Practices

1. **Be Specific in Prompts**: Tell LLM exactly what format you want
2. **Provide Examples**: Show example output in your prompt
3. **Clean the Output**: LLMs sometimes add extra text - strip it
4. **Validate Parsing**: Always check if parsing succeeded
5. **Error Handling**: Use try-except blocks to catch parsing errors

## Which Format to Use?

- **String**: Simple answers, natural language
- **JSON**: Complex nested data, structured objects
- **CSV**: Tabular data, lists of records
- **List**: Multiple items of same type
- **Key-Value**: Simple properties and attributes
- **YAML**: Configuration-like data, hierarchical

## Next Steps

Once you understand these basics, you can:
- Combine multiple formats
- Create custom parsers
- Use Pydantic models for validation (advanced)
- Integrate with databases
- Build data pipelines
