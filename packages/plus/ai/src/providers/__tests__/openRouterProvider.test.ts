import * as assert from 'assert';
import { openRouterProviderDescriptor } from '../../constants.js';
import type { AIModel } from '../../models/model.js';
import type { AIChatMessage, AIChatMessageRole, AIResponseFormat } from '../../models/provider.js';
import type { AIOpenRouterRouting, AIProviderContext } from '../context.js';
import { OpenRouterProvider } from '../openRouterProvider.js';
import { clearResponseFormatRejections } from '../responseFormatCache.js';
import { createTransport } from './fixtures.js';

/**
 * OpenRouter is the only provider that carries user-authored routing on the request. Two things can
 * silently erase it — `applyResponseFormat` overwriting the block, and the base's format retry
 * deleting it — and both fail invisibly, as a request that simply routes somewhere else. These pin
 * both, plus the preset listing that turns a user's named configurations into pickable models.
 */

const model: AIModel<'openrouter'> = {
	id: 'anthropic/claude-sonnet-5',
	name: 'Claude Sonnet 5',
	maxTokens: { input: 200000, output: 64000 },
	provider: openRouterProviderDescriptor,
	temperature: null,
	supportsStructuredOutputs: true,
};

const messages: AIChatMessage<AIChatMessageRole>[] = [{ role: 'user', content: 'Summarize this diff' }];

const responseFormat: AIResponseFormat = {
	name: 'summary',
	schema: { type: 'object', properties: { summary: { type: 'string' } } },
};

const okResponse = {
	id: 'r1',
	model: model.id,
	choices: [{ index: 0, message: { role: 'assistant', content: 'ok' } }],
};

function routingContext(routing: AIOpenRouterRouting | undefined, responses: unknown[]) {
	return createTransport(responses, { getOpenRouterRouting: () => routing });
}

function send(context: AIProviderContext, options?: { responseFormat?: AIResponseFormat }) {
	return new OpenRouterProvider(context).sendRequest(
		'generate-commitMessage',
		model,
		'test-key',
		() => Promise.resolve(messages),
		{ signal: new AbortController().signal, responseFormat: options?.responseFormat },
	);
}

suite('OpenRouterProvider routing Test Suite', () => {
	setup(() => clearResponseFormatRejections());

	test('sends no provider block when no routing is configured', async () => {
		const { context, sent } = routingContext(undefined, [okResponse]);
		await send(context);

		assert.strictEqual('provider' in sent[0].body, false);
	});

	test('translates routing settings to the API vocabulary, omitting what is unset', async () => {
		const { context, sent } = routingContext({ sort: 'throughput', only: ['anthropic'], allowFallbacks: false }, [
			okResponse,
		]);
		await send(context);

		assert.deepStrictEqual(sent[0].body.provider, {
			sort: 'throughput',
			only: ['anthropic'],
			allow_fallbacks: false,
		});
	});

	test('the response format merges into routing instead of replacing it', async () => {
		const { context, sent } = routingContext({ sort: 'throughput' }, [okResponse]);
		await send(context, { responseFormat: responseFormat });

		assert.deepStrictEqual(sent[0].body.provider, { sort: 'throughput', require_parameters: true });
	});

	test('the format retry drops require_parameters but keeps the user routing', async () => {
		// OpenRouter answers 404 when `require_parameters` matches no serving endpoint; the retry has
		// to strip that constraint alone — deleting the whole block would silently reroute the request
		const { context, sent } = routingContext({ sort: 'throughput', only: ['anthropic'] }, [
			{ status: 404, body: { error: { message: 'No endpoints found matching your parameters' } } },
			okResponse,
		]);
		await send(context, { responseFormat: responseFormat });

		assert.strictEqual(sent.length, 2);
		assert.strictEqual('response_format' in sent[1].body, false);
		assert.deepStrictEqual(sent[1].body.provider, { sort: 'throughput', only: ['anthropic'] });
	});

	test('the format retry removes an empty provider block entirely', async () => {
		const { context, sent } = routingContext(undefined, [
			{ status: 404, body: { error: { message: 'No endpoints found matching your parameters' } } },
			okResponse,
		]);
		await send(context, { responseFormat: responseFormat });

		assert.strictEqual('provider' in sent[1].body, false);
	});
});

suite('OpenRouterProvider presets Test Suite', () => {
	const catalog = {
		data: [
			{
				id: 'anthropic/claude-sonnet-5',
				name: 'Claude Sonnet 5',
				context_length: 200000,
				top_provider: { max_completion_tokens: 64000 },
				supported_parameters: ['structured_outputs'],
			},
		],
	};

	test('lists presets ahead of the catalog, with limits from the pinned model', async () => {
		const { context } = createTransport([
			catalog,
			{
				data: [
					{
						id: 'p_1',
						slug: 'fast-commits',
						name: 'Fast commits',
						designated_version: { config: { model: 'anthropic/claude-sonnet-5' } },
					},
				],
			},
		]);

		const models = await new OpenRouterProvider(context).getModels();

		assert.strictEqual(models[0].id, '@preset/fast-commits');
		assert.strictEqual(models[0].name, 'Fast commits (preset)');
		assert.deepStrictEqual(models[0].maxTokens, { input: 200000, output: 64000 });
		assert.strictEqual(models[0].supportsStructuredOutputs, true);
		assert.strictEqual(models[1].id, 'anthropic/claude-sonnet-5');
	});

	test('falls back to a conservative window when the preset pins no known model', async () => {
		const { context } = createTransport([catalog, { data: [{ slug: 'router', name: 'Router' }] }]);

		const models = await new OpenRouterProvider(context).getModels();

		assert.deepStrictEqual(models[0].maxTokens, { input: 128000, output: 32768 });
		// A preset can route anywhere, so it must not inherit a schema-support claim
		assert.strictEqual(models[0].supportsStructuredOutputs, false);
	});

	test('parses a bare array listing, without the data envelope', async () => {
		const { context } = createTransport([catalog, [{ slug: 'bare', name: 'Bare' }]]);

		const models = await new OpenRouterProvider(context).getModels();

		assert.strictEqual(models[0].id, '@preset/bare');
	});

	test('a key without preset scope still yields the full catalog', async () => {
		const { context } = createTransport([catalog, { status: 403, body: { error: 'forbidden' } }]);

		const models = await new OpenRouterProvider(context).getModels();

		assert.strictEqual(models.length, 1);
		assert.strictEqual(models[0].id, 'anthropic/claude-sonnet-5');
	});
});
