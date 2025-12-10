# Database Migrations

This folder contains database schema and migrations for the Supabase database.

## Quick Setup (SQL Editor Method)

The easiest way to set up the database is to use Supabase's SQL Editor:

1. Go to your Supabase Dashboard
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `migrations/20240101000000_initial_schema.sql`
4. Click **Run**

## Using Supabase CLI (Optional)

You can use the Supabase CLI for more advanced migration management. There are several ways to use it:

### Method 1: Use npx (No Installation Required)

```bash
# Run commands directly with npx
npx supabase --help
npx supabase link --project-ref your-project-ref
npx supabase db push
```

**Note:** Requires Node.js 20 or later.

### Method 2: Install as Dev Dependency

```bash
# Install in your project
npm install supabase --save-dev

# Then run commands
npx supabase link --project-ref your-project-ref
npx supabase db push
```

### Method 3: Install via Scoop (Windows)

```bash
# Using Scoop package manager
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase

# Then use directly
supabase link --project-ref your-project-ref
supabase db push
```

### Using the CLI

Once set up, you can:

```bash
# Link to your Supabase project
supabase link --project-ref your-project-ref

# Push migrations to your database
supabase db push

# Create a new migration
supabase migration new your_migration_name
```

**Important:** `npm install -g supabase` (global install) is NOT supported. Use one of the methods above instead.

## Migration Files

- `migrations/20240101000000_initial_schema.sql` - Initial schema with tables and indexes
- `migrations/20240110000000_add_conversation_to_chunks.sql` - Adds conversation_id to document_chunks for conversation isolation

## Schema Overview

- **document_chunks**: Stores PDF text chunks with vector embeddings
- **conversations**: Stores conversation metadata
- **messages**: Stores individual messages within conversations

