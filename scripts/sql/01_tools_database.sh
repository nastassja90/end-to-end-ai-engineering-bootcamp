#!/bin/bash

# Create the new tools database only if it does not exist
# \gexec allows us to execute the result of the SELECT statement as a new query
# this is a common pattern for conditionally creating databases if they don't exist
    
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE tools_database'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'tools_database')\gexec
EOSQL