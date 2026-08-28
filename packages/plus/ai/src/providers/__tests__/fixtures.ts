import type { AIProviderContext } from '../context.js';

/** Inert provider context for construction-only tests — nothing under test may reach it */
export function createStubProviderContext(overrides?: Partial<AIProviderContext>): AIProviderContext {
	return {
		fetch: () => Promise.reject(new Error('not used')),
		getApiKey: () => Promise.resolve(undefined),
		getProviderConfig: () => ({ enabled: true }),
		getOrPromptUrl: () => Promise.resolve(undefined),
		...overrides,
	};
}

/** A queued response: a bare body replays as 200, the object form drives the status too */
export type TransportResponse = unknown | { status: number; body: unknown };

export interface CapturedRequest {
	url: string;
	headers: Record<string, string>;
	body: Record<string, unknown>;
}

/**
 * Captures each outgoing request and replays queued responses in order, so a provider can be driven
 * through its real `sendRequest` → `fetchCore` path with no API key and no network.
 */
export function createTransport(
	responses: TransportResponse[],
	overrides?: Partial<AIProviderContext>,
): { context: AIProviderContext; sent: CapturedRequest[] } {
	const sent: CapturedRequest[] = [];
	let call = 0;

	const context: AIProviderContext = {
		fetch: (url: string | URL, init?: RequestInit) => {
			sent.push({
				url: String(url),
				headers: (init?.headers ?? {}) as Record<string, string>,
				body: JSON.parse(typeof init?.body === 'string' ? init.body : '{}') as Record<string, unknown>,
			});

			const queued = responses[Math.min(call++, responses.length - 1)];
			const framed =
				queued != null && typeof queued === 'object' && 'status' in queued && 'body' in queued
					? (queued as { status: number; body: unknown })
					: { status: 200, body: queued };

			return Promise.resolve(new Response(JSON.stringify(framed.body), { status: framed.status }));
		},
		getApiKey: () => Promise.resolve('test-key'),
		getProviderConfig: () => ({ enabled: true }),
		getOrPromptUrl: () => Promise.resolve(undefined),
		...overrides,
	};

	return { context: context, sent: sent };
}
