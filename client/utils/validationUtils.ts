/**
 * Utility functions for input validation
 */

/**
 * Validates a query string
 * @param query The query to validate
 * @returns Object containing validation result and error message
 */
export function validateQuery(query: string): { isValid: boolean; error?: string } {
    if (!query.trim()) {
        return { isValid: false, error: 'Query cannot be empty' };
    }

    if (query.length < 3) {
        return { isValid: false, error: 'Query must be at least 3 characters long' };
    }

    if (query.length > 500) {
        return { isValid: false, error: 'Query cannot exceed 500 characters' };
    }

    return { isValid: true };
}

/**
 * Sanitizes a string input
 * @param input The string to sanitize
 * @returns Sanitized string
 */
export function sanitizeInput(input: string): string {
    return input.trim().replace(/[<>]/g, '');
}