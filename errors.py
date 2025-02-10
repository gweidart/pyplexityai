"""Error types for Perplexity AI client."""

from diagnostic import DiagnosticError


class PerplexityError(DiagnosticError):
    """Base error class for all Perplexity client errors."""

    docs_index = "https://github.com/gweidart/pyplexityai/blob/main/README.md#{code}"


class AuthenticationError(PerplexityError):
    def __init__(self, api_key: str):
        """
        Initialize an AuthenticationError instance.

        This constructor creates an AuthenticationError with predefined error details
        related to API key authentication failures with Perplexity AI.

        Parameters
        ----------
        api_key (str): The API key that failed authentication. This is used to provide
                       a partially masked version of the key in the error message.

        Returns
        -------
        None: This method initializes the object and doesn't return anything.

        """
        super().__init__(
            code="auth-001",
            message="Failed to authenticate with Perplexity AI",
            causes=[
                "Invalid or expired API key provided",
                f"API key used: {api_key[:6]}...{api_key[-4:]}"
                if api_key
                else "No API key provided",
            ],
            hint_stmt="Please check your API key and ensure it is valid.",
            note_stmt="You can find your API key in your Perplexity AI dashboard.",
        )


class SearchError(PerplexityError):
    def __init__(self, query: str, mode: str, error: Exception | None = None):
        """
        Initialize a SearchError instance.

        This constructor creates a SearchError with predefined error details
        related to failed search requests in Perplexity AI.

        Parameters
        ----------
        query : str
            The search query that failed to execute.
        mode : str
            The search mode that was used for the query.
        error : Exception | None, optional
            The exception that was raised during the search, if any.

        Returns
        -------
        None
            This method initializes the object and doesn't return anything.

        """
        causes = [
            f"Failed to execute search query: {query!r}",
            f"Search mode used: {mode}",
        ]
        if error:
            causes.append(f"Error details: {error!s}")

        super().__init__(
            code="search-001",
            message="Search request failed",
            causes=causes,
            hint_stmt="Try simplifying your query or using a different search mode.",
            note_stmt="Valid search modes are 'concise' and 'copilot'.",
        )


class PerplexityTimeoutError(PerplexityError):
    def __init__(self, query: str, timeout: float):
        """
        Initialize a PerplexityTimeoutError instance.

        This constructor creates a PerplexityTimeoutError with predefined error details
        related to search requests that have timed out in Perplexity AI.

        Parameters
        ----------
        query : str
            The search query that timed out.
        timeout : float
            The duration in seconds after which the request timed out.

        Returns
        -------
        None
            This method initializes the object and doesn't return anything.

        """
        super().__init__(
            code="timeout-001",
            message="Search request timed out",
            causes=[f"Query: {query!r}", f"Timeout after {timeout} seconds"],
            hint_stmt="Try increasing the timeout value or simplifying your query.",
            note_stmt="Complex queries may take longer to process.",
        )


class InvalidParameterError(PerplexityError):
    def __init__(self, param: str, value: str, valid_values: list[str] | None = None):
        """
        Initialize an InvalidParameterError instance.

        This constructor creates an InvalidParameterError with predefined error details
        related to invalid parameter values in Perplexity AI requests.

        Parameters
        ----------
        param : str
            The name of the parameter that has an invalid value.
        value : str
            The invalid value that was provided for the parameter.
        valid_values : list[str] | None, optional
            A list of valid values for the parameter, if applicable.

        Returns
        -------
        None
            This method initializes the object and doesn't return anything.

        """
        causes = [f"Invalid value for parameter {param!r}: {value!r}"]
        if valid_values:
            causes.append(f"Valid values are: {', '.join(valid_values)}")

        super().__init__(
            code="param-001",
            message=f"Invalid parameter: {param}",
            causes=causes,
            hint_stmt=f"Please provide a valid value for {param}.",
            note_stmt="Check the documentation for allowed parameter values.",
        )
