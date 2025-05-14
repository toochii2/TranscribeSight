# Contributing to TranscribeSight

Thank you for considering contributing to TranscribeSight! This document provides guidelines for contributing.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Check the issue tracker to see if the bug has already been reported
- If not, create a new issue with a clear title and description
- Include as much relevant information as possible
- Add steps to reproduce the issue
- Include screenshots if applicable

### Suggesting Enhancements

- Check the issue tracker to see if the enhancement has already been suggested
- If not, create a new issue with a clear title and description
- Describe the current behavior and explain the behavior you expected to see
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure your code follows the existing style
4. Issue a pull request

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Adding New Service Providers

To add support for a new speech-to-text service:

1. Create a new function in `app.py` for the service:
   ```python
   def transcribe_new_service(audio_path, client):
       """
       Transcribe audio using the new service.
       Returns text and transcription time.
       """
       # Implementation
       return result_text, transcription_time
   ```

2. Add the service to the `SERVICE_COSTS` dictionary
3. Update the UI in the sidebar to include the new service
4. Add the service to the processing logic

## Styling Guidelines

- Follow PEP 8
- Use docstrings for all functions and classes
- Keep line length to 100 characters or less
- Use descriptive variable names

## Testing

- Include tests for new functionality
- Ensure all tests pass before submitting a pull request
- Include test files for different speech patterns if relevant

## Documentation

- Update the README.md if you change functionality
- Comment your code where necessary
- Update the docstrings for any functions you modify

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.