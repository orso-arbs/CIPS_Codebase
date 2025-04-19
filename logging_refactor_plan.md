# Logging Refactor Plan: Python `logging` Integration

## Objective

Replace the custom logging implementation in `Format_1.py` (specifically the `@ParameterLog` decorator and `save_args_to_json` function) with Python's standard `logging` library. The goal is to maintain the existing functionality:
1.  Log function arguments to a `_log.json` file within the function's output directory.
2.  Preserve the chained logging behavior where each step reads the log from its `input_dir`, prepends its own parameters, and writes the updated log to its `output_dir`.
3.  Retain the existing directory structure generation (`script_name_YYYY-MM-DD_HH-MM-SS_comment`).
4.  Continue truncating large argument values before logging.

## Analysis Summary

*   **Directory Structure:** `Format_1.F_out_dir` handles the creation of timestamped output directories based on script name, date, and optional comment. This function should be retained.
*   **Current Logging:** The `@F_1.ParameterLog` decorator intercepts function calls, processes arguments (using `truncate_value` and `deep_sizeof`), executes the function, retrieves `input_dir` and `output_dir`, and calls `save_args_to_json`.
*   **`save_args_to_json`:** Reads `_log.json` from `input_dir`, prepends the current function's processed arguments (as `{function_name: args_dict}`), and writes the combined JSON to `output_dir/_log.json`.
*   **Workflow:** Scripts like `CP_main_1.py` chain operations, passing the `output_dir` of one step as the `input_dir` to the next, propagating the `_log.json` file.

## Proposed Plan Steps

1.  **Retain Directory Logic:** Keep `Format_1.F_out_dir` as is.
2.  **Implement Custom Logging Handler:**
    *   Create a `logging.Handler` subclass (e.g., `JsonChainHandler`) in `Format_1.py` or a dedicated logging utility file.
    *   This handler will be configured *per function call* with the specific `input_dir` and `output_dir`.
    *   Its `emit(record)` method will perform the core logic:
        *   Get `input_dir` and `output_dir` from its configuration.
        *   Construct input and output log file paths (`.../_log.json`).
        *   Read JSON data from the input log file (handle `FileNotFoundError`, `JSONDecodeError`).
        *   Format the new log record's data (passed via `record.msg` or similar) into the `{function_name: processed_args_dict}` structure.
        *   Prepend this new dictionary to the loaded data.
        *   Write the updated JSON data to the output log file (overwrite).
3.  **Implement Custom Logging Formatter (Recommended):**
    *   Create a `logging.Formatter` subclass (e.g., `ArgTruncatingFormatter`).
    *   Its `format(record)` method will:
        *   Take the raw arguments dictionary (passed via `record.msg` or `record.args`).
        *   Apply the `Format_1.truncate_value` logic to each argument.
        *   Return the processed arguments dictionary, ready for the handler.
4.  **Create New Decorator:**
    *   Implement a new decorator (e.g., `@StandardLog`) in `Format_1.py`.
    *   This decorator will:
        *   Inspect function arguments (`*args`, `**kwargs`).
        *   Extract `input_dir` from arguments.
        *   Execute the wrapped function to get the `output_dir`.
        *   Create a temporary `logging.Logger` instance (or get a uniquely named logger).
        *   Create instances of the `JsonChainHandler` (configured with `input_dir`, `output_dir`) and `ArgTruncatingFormatter`.
        *   Add the handler and formatter to the logger. Set level appropriately. Ensure handlers are removed after logging to prevent duplication if the logger is reused.
        *   Prepare the arguments dictionary (without truncation, as the formatter handles it).
        *   Log the arguments dictionary: `logger.info(arguments_dict)`.
        *   Crucially, ensure the logger/handler setup is isolated per decorated function call to correctly associate `input_dir` and `output_dir`.
5.  **Refactor Code:**
    *   Replace all occurrences of `@F_1.ParameterLog` with the new `@StandardLog` decorator across all relevant project files (e.g., `CP_segment_1.py`, `CP_extract_1.py`, `Visit_Projector_1.py`, etc.).
    *   Remove the old `ParameterLog` decorator and the `save_args_to_json` function from `Format_1.py`.

## Visual Plan (Mermaid)

```mermaid
graph TD
    A[Function Call with @StandardLog] --> B{Decorator Executes};
    B --> C[Extract input_dir];
    B --> D[Prepare Raw Arguments Dict];
    B --> E[Execute Original Function];
    E --> F[Function Returns output_dir];
    B --> G{Configure Logger};
    G --> H[Instantiate & Configure JsonChainHandler (with input/output dirs)];
    G --> I[Instantiate ArgTruncatingFormatter];
    G --> L[Add Handler & Formatter to Logger];
    C --> H;
    F --> H;
    D --> J[Log Raw Arguments Dict];
    L --> J;
    J --> K{Formatter Processes Record};
    K -- Processed Args --> M{Handler Emits Record};
    M --> N[Read input_dir/_log.json];
    M --> O[Format New Log Entry Dict];
    N & O --> P[Prepend New Entry to Old Data];
    P --> Q[Write output_dir/_log.json];

    style M fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#ccf,stroke:#333,stroke-width:2px
```

## Key Challenges & Considerations

*   **Handler Configuration:** Ensuring the `JsonChainHandler` gets the correct `input_dir` and `output_dir` for each specific function call is critical. Passing them during instantiation within the decorator seems most direct.
*   **Logger Isolation:** Avoid handlers accumulating on logger instances across calls. Creating temporary loggers or carefully managing handler addition/removal is necessary.
*   **Error Handling:** Implement robust error handling for file I/O and JSON operations within the handler.
*   **Argument Processing:** Ensure `truncate_value` is correctly applied, ideally within the formatter.