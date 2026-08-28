import * as assert from 'assert';
import { omniRouteProviderDescriptor } from '../../constants.js';
import type { AIActionType, AIModel } from '../../models/model.js';
import type { AIProviderContext } from '../context.js';
import { OmniRouteProvider } from '../omniRouteProvider.js';
import { createStubProviderContext, createTransport } from './fixtures.js';

/**
 * OmniRoute is the one provider whose key is genuinely optional: a fresh self-hosted gateway answers
 * unauthenticated. These lock the two behaviors that follow from that — no blocking key prompt ever,
 * and no `Authorization` header when there is no real key — plus the dynamic catalog it is here for.
 */

const model: AIModel<'omniroute'> = {
	id: 'auto/best-coding',
	name: 'Auto (best coding)',
	maxTokens: { input: 128000, output: 64000 },
	provider: omniRouteProviderDescriptor,
};

/** Exposes the `protected` hooks under test without loosening them on the provider itself */
class TestOmniRouteProvider extends OmniRouteProvider {
	getUrlForTest(m: AIModel<'omniroute'>): string | undefined {
		return this.getUrl(m);
	}

	getHeadersForTest(apiKey: string): Record<string, string> {
		return this.getHeaders('generate-commitMessage', apiKey, model, 'https://example.com');
	}
}

function createProvider(context: AIProviderContext): TestOmniRouteProvider {
	return new TestOmniRouteProvider(context);
}

suite('OmniRouteProvider Test Suite', () => {
	test('getUrl falls back to the default gateway when no URL is configured', () => {
		const provider = createProvider(createStubProviderContext());

		assert.strictEqual(provider.getUrlForTest(model), 'http://localhost:20128/v1/chat/completions');
	});

	test('getUrl uses the configured URL', () => {
		const provider = createProvider(
			createStubProviderContext({ getProviderConfig: () => ({ enabled: true, url: 'http://gateway:9000/v1' }) }),
		);

		assert.strictEqual(provider.getUrlForTest(model), 'http://gateway:9000/v1/chat/completions');
	});

	test('getUrl is undefined when the provider is disabled', () => {
		const provider = createProvider(createStubProviderContext({ getProviderConfig: () => ({ enabled: false }) }));

		assert.strictEqual(provider.getUrlForTest(model), undefined);
	});

	test('getHeaders omits Authorization for the unauthenticated sentinel', () => {
		const provider = createProvider(createStubProviderContext());

		const headers = provider.getHeadersForTest('<not applicable>');
		assert.strictEqual(headers.Authorization, undefined);
		assert.strictEqual(headers.Accept, 'application/json');
	});

	test('getHeaders omits Authorization for an empty key', () => {
		const provider = createProvider(createStubProviderContext());

		assert.strictEqual(provider.getHeadersForTest('').Authorization, undefined);
	});

	test('getHeaders sends Authorization for a real key', () => {
		const provider = createProvider(createStubProviderContext());

		assert.strictEqual(provider.getHeadersForTest('omni-key').Authorization, 'Bearer omni-key');
	});

	test('getApiKey never prompts, even when asked non-silently', async () => {
		// Rejects on any non-silent read — a prompt would be a blocking gate on a provider that may
		// need no key at all
		const provider = createProvider(
			createStubProviderContext({
				getApiKey: (_config, silent) =>
					silent ? Promise.resolve(undefined) : Promise.reject(new Error('prompted')),
			}),
		);

		assert.strictEqual(await provider.getApiKey(false), '<not applicable>');
	});

	test('getApiKey returns a stored key when the context has one', async () => {
		const provider = createProvider(createStubProviderContext({ getApiKey: () => Promise.resolve('stored-key') }));

		assert.strictEqual(await provider.getApiKey(false), 'stored-key');
	});

	test('getModels maps the OpenAI-shaped catalog', async () => {
		const { context, sent } = createTransport([
			{ data: [{ id: 'openai/gpt-5', name: 'GPT-5', context_length: 400000 }, { id: 'auto/best-free' }] },
		]);
		const provider = createProvider(context);

		const models = await provider.getModels();

		assert.strictEqual(sent[0].url, 'http://localhost:20128/v1/models');
		assert.strictEqual(models.length, 2);
		assert.strictEqual(models[0].name, 'GPT-5');
		assert.strictEqual(models[0].maxTokens.input, 400000);
		assert.strictEqual(models[0].maxTokens.output, 200000);
		// Name falls back to the id, and an absent context_length to a conservative window
		assert.strictEqual(models[1].name, 'auto/best-free');
		assert.strictEqual(models[1].maxTokens.input, 128000);
	});

	test('getModels returns empty on a failed response', async () => {
		const { context } = createTransport([{ status: 500, body: { error: 'boom' } }]);
		const provider = createProvider(context);

		assert.deepStrictEqual(await provider.getModels(), []);
	});

	test('getModels returns empty when the gateway is unreachable', async () => {
		const provider = createProvider(
			createStubProviderContext({ fetch: () => Promise.reject(new Error('ECONNREFUSED')) }),
		);

		assert.deepStrictEqual(await provider.getModels(), []);
	});

	test('configured is true when the gateway answers, false when it does not', async () => {
		const { context } = createTransport([{ data: [] }]);
		assert.strictEqual(await createProvider(context).configured(true), true);

		const unreachable = createStubProviderContext({ fetch: () => Promise.reject(new Error('ECONNREFUSED')) });
		assert.strictEqual(await createProvider(unreachable).configured(true), false);
	});
});
