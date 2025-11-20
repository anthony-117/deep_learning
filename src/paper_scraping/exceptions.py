class ScraperException(Exception):
    pass


class SourceNotFoundError(ScraperException):
    def __init__(self, source: str, available_sources: list[str] = None):
        self.source = source
        self.available_sources = available_sources or []
        message = f"Scraper source '{source}' not found."
        if self.available_sources:
            message += f" Available sources: {', '.join(self.available_sources)}"
        super().__init__(message)


class RateLimitError(ScraperException):
    def __init__(self, source: str, retry_after: float = None):
        self.source = source
        self.retry_after = retry_after
        message = f"Rate limit exceeded for '{source}'."
        if retry_after:
            message += f" Retry after {retry_after:.1f} seconds."
        super().__init__(message)


class DownloadError(ScraperException):
    def __init__(self, paper_id: str, source: str, reason: str = None):
        self.paper_id = paper_id
        self.source = source
        self.reason = reason
        message = f"Failed to download paper '{paper_id}' from '{source}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message)


class InvalidQueryError(ScraperException):
    def __init__(self, query: str, reason: str = None):
        self.query = query
        self.reason = reason
        message = f"Invalid query: '{query}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message)


class APIError(ScraperException):
    def __init__(self, source: str, status_code: int = None, message: str = None):
        self.source = source
        self.status_code = status_code
        error_message = f"API error for '{source}'."
        if status_code:
            error_message += f" Status code: {status_code}."
        if message:
            error_message += f" Message: {message}"
        super().__init__(error_message)


class ConfigurationError(ScraperException):
    pass