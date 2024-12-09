/**
 * Utility functions for file operations
 */

/**
 * Formats a file size in bytes to a human-readable string
 * @param bytes The size in bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Validates a file type against allowed extensions
 * @param file The file to validate
 * @param allowedTypes Array of allowed file extensions
 * @returns boolean indicating if the file type is allowed
 */
export function isValidFileType(file: File, allowedTypes: string[]): boolean {
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    return allowedTypes.includes(extension);
}

/**
 * Checks if a file size is within the allowed limit
 * @param file The file to check
 * @param maxSize Maximum allowed size in bytes
 * @returns boolean indicating if the file size is allowed
 */
export function isValidFileSize(file: File, maxSize: number): boolean {
    return file.size <= maxSize;
}