# Define reusable logic once
backup_files() {
    local source_dir=$1
    local backup_dir=$2
    
    echo "Creating backup..."
    cp -r "$source_dir" "$backup_dir/backup_$(date +%Y%m%d)"
    echo "Backup complete!"
}

# Reuse the logic multiple times
backup_files "/home/user/documents" "/backups"
backup_files "/home/user/photos" "/backups"  
backup_files "/var/www" "/backups"
