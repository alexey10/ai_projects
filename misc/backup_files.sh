#!/bin/bash

# Backup Script for Local Environment
# Usage: ./backup.sh or source this file to use the function

backup_files() {
    local source_dir=$1
    local backup_dir=$2
    
    # Check if both arguments are provided
    if [ -z "$source_dir" ] || [ -z "$backup_dir" ]; then
        echo "Usage: backup_files <source_directory> <backup_directory>"
        echo "Example: backup_files ~/Documents ~/Backups"
        return 1
    fi
    
    # Check if source directory exists
    if [ ! -d "$source_dir" ]; then
        echo "Error: Source directory '$source_dir' does not exist!"
        return 1
    fi
    
    # Create backup directory if it doesn't exist
    if [ ! -d "$backup_dir" ]; then
        echo "Creating backup directory: $backup_dir"
        mkdir -p "$backup_dir"
    fi
    
    # Generate timestamp for unique backup name
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_name="backup_$(basename "$source_dir")_$timestamp"
    full_backup_path="$backup_dir/$backup_name"
    
    echo "🔄 Creating backup..."
    echo "   Source: $source_dir"
    echo "   Destination: $full_backup_path"
    
    # Perform the backup
    if cp -r "$source_dir" "$full_backup_path"; then
        echo "✅ Backup complete!"
        echo "   Backup saved as: $backup_name"
        
        # Show backup size
        backup_size=$(du -sh "$full_backup_path" | cut -f1)
        echo "   Backup size: $backup_size"
    else
        echo "❌ Backup failed!"
        return 1
    fi
}

# Function to list existing backups
list_backups() {
    local backup_dir=${1:-~/Backups}
    
    if [ ! -d "$backup_dir" ]; then
        echo "No backup directory found at: $backup_dir"
        return 1
    fi
    
    echo "📁 Existing backups in $backup_dir:"
    ls -la "$backup_dir" | grep "^d" | grep "backup_" | awk '{print "   " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
}

# Function to clean old backups (keep only last N backups)
cleanup_old_backups() {
    local backup_dir=$1
    local keep_count=${2:-5}  # Keep last 5 backups by default
    
    if [ ! -d "$backup_dir" ]; then
        echo "Backup directory '$backup_dir' does not exist!"
        return 1
    fi
    
    echo "🧹 Cleaning up old backups (keeping last $keep_count)..."
    
    # Find and remove old backups
    ls -1t "$backup_dir"/backup_* 2>/dev/null | tail -n +$((keep_count + 1)) | while read -r old_backup; do
        echo "   Removing: $(basename "$old_backup")"
        rm -rf "$old_backup"
    done
    
    echo "✅ Cleanup complete!"
}

# Example usage function
show_examples() {
    echo "📖 Example Usage:"
    echo ""
    echo "1. Backup your Documents folder:"
    echo "   backup_files ~/Documents ~/Backups"
    echo ""
    echo "2. Backup a specific project:"
    echo "   backup_files ~/Projects/my-app ~/Backups"
    echo ""
    echo "3. List existing backups:"
    echo "   list_backups ~/Backups"
    echo ""
    echo "4. Clean old backups (keep last 3):"
    echo "   cleanup_old_backups ~/Backups 3"
    echo ""
    echo "💡 Tips:"
    echo "   - Use full paths or ~ for home directory"
    echo "   - Backup directory will be created if it doesn't exist"
    echo "   - Each backup gets a unique timestamp"
}

# Main execution when script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "🔧 Backup Script Loaded!"
    echo ""
    
    # Show examples if no arguments provided
    if [ $# -eq 0 ]; then
        show_examples
        echo ""
        echo "Available functions:"
        echo "  - backup_files <source> <destination>"
        echo "  - list_backups [backup_directory]"
        echo "  - cleanup_old_backups <backup_directory> [keep_count]"
        echo "  - show_examples"
    else
        # If arguments provided, try to run backup_files
        backup_files "$1" "$2"
    fi
fi
