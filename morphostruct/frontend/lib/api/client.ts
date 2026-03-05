const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000]; // milliseconds
const REQUEST_TIMEOUT = 30000; // 30 seconds

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isRetryableError(error: unknown): boolean {
  if (error instanceof ApiError) {
    // Retry on 5xx server errors, not 4xx client errors
    return error.status >= 500 && error.status < 600;
  }
  // Retry on network failures (TypeError for fetch failures)
  if (error instanceof TypeError) {
    return true;
  }
  return false;
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number
): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Request timeout')), timeoutMs);
  });
  return Promise.race([promise, timeoutPromise]);
}

async function retryWrapper<T>(
  fn: () => Promise<T>,
  maxRetries: number = MAX_RETRIES,
  timeoutMs: number = REQUEST_TIMEOUT
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await withTimeout(fn(), timeoutMs);
    } catch (error) {
      lastError = error;

      // Don't retry if it's not a retryable error
      if (!isRetryableError(error)) {
        throw error;
      }

      // Don't retry if we've exhausted attempts
      if (attempt === maxRetries) {
        throw error;
      }

      // Wait before retrying
      const delay = RETRY_DELAYS[attempt] || RETRY_DELAYS[RETRY_DELAYS.length - 1];
      await sleep(delay);
    }
  }

  throw lastError;
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = REQUEST_TIMEOUT
): Promise<T> {
  return retryWrapper(async () => {
    const url = `${API_BASE_URL}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        errorData.detail || `HTTP error ${response.status}`
      );
    }

    return response.json();
  }, MAX_RETRIES, timeoutMs);
}

export async function apiBlobRequest(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = REQUEST_TIMEOUT
): Promise<Blob> {
  return retryWrapper(async () => {
    const url = `${API_BASE_URL}${endpoint}`;

    const response = await fetch(url, options);

    if (!response.ok) {
      // Try to parse error message from JSON first, fall back to text
      let errorMessage = `HTTP error ${response.status}`;
      try {
        const errorData = await response.clone().json();
        errorMessage = errorData.detail || errorMessage;
      } catch {
        // If JSON parsing fails, try to get text
        try {
          const errorText = await response.text();
          if (errorText) {
            errorMessage = errorText;
          }
        } catch {
          // Keep default error message
        }
      }
      throw new ApiError(response.status, errorMessage);
    }

    return response.blob();
  }, MAX_RETRIES, timeoutMs);
}

