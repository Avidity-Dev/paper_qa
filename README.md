# Development Setup Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+

## Database Setup

The project uses Redis Stack for vector storage and PostgreSQL for relational data. Both services are configured in the Docker Compose file.

To start all services in detached mode:

```bash
docker compose up -d
```

This will start the Redis Stack and PostgreSQL services. The local data should be persisted
throughout restarts.

### Verify the services are running

```bash
docker compose ps
```

This will show the status of all services.

### To stop the services

```bash
docker compose down
```

## Using CLI `manage.py` tool

The `manage.py` tool is a CLI tool for managing the Redis search indexes and documents.

To see the available commands, run:

```bash
python manage.py
```

To see the help for a specific command, run:

```bash
python manage.py <command> --help
# example: python manage.py create-index --help
```

### Current Commands

- `create-index`: Create a new Redis search index using the specified configuration.
- `delete-index`: Delete an existing Redis search index, with the option to delete the data as well.
- `clear-documents`: Clear all documents from a Redis search index.
- `populate-test-data`: Populate the Redis search index with test data. **NOT IMPLEMENTED YET.**

## Development Environment Setup

1. Create a new Python virtual environment
2. Install the dependencies
3. Spin up docker services
4. Run the CLI tool to create the Redis index

## Application Configuration

The application configuration settings can be found in the `src/config/` directory.

### Configuration Files

- `app.yaml`: Environment-specific application settings
- `static.yaml`: Model configurations and endpoints used to determine service configs based off the application environment.
- `vector.yaml`: Redis vector store schema configuration.

### Environment Variables

The following environment variables are used to locate configuration files:

```bash
REDIS_VECTOR_CONFIG_PATH=src/config/vector.yaml
APP_CONFIG_PATH=src/config/app.yaml
STATIC_CONFIG_PATH=src/config/static.yaml
```

API keys may need to be set in the environment variables, depending on the provider.
Example:

```bash
OPENAI_API_KEY=sk-...
```


