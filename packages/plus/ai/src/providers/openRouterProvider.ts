import { isCancellationError } from '@gitlens/utils/cancellation.js';
import { openRouterProviderDescriptor as provider } from '../constants.js';
import type { AIActionType, AIModel } from '../models/model.js';
import type { AIResponseFormat } from '../models/provider.js';
import type { ChatCompletionRequest } from './openAICompatibleProviderBase.js';
import { OpenAICompatibleProviderBase } from './openAICompatibleProviderBase.js';

type OpenRouterModel = AIModel<typeof provider.id>;

export class OpenRouterProvider extends OpenAICompatibleProviderBase<typeof provider.id> {
	readonly id = provider.id;
	readonly name = provider.name;
	readonly supportsTools = true;
	protected readonly descriptor = provider;
	protected readonly config = {
		keyUrl: 'https://openrouter.ai/keys',
		keyValidator: /(?:sk-)?\w{24,128}/,
	};

	async getModels(): Promise<readonly AIModel<typeof provider.id>[]> {
		let apiKey: string | undefined;
		try {
			apiKey = await this.getApiKey(true);
		} catch (ex) {
			if (isCancellationError(ex)) return [];

			throw ex;
		}

		if (!apiKey) return [];

		const headers = this.getHeadersCore(apiKey);
		const models = await this.getCatalogModels(headers);
		// A key without preset scope 403s here; presets are an enhancement, never a reason to
		// strand the user on an empty model list
		const presets = await this.getPresetModels(headers, models);

		// Presets lead: they're the user's own named configurations, and the catalog is hundreds long
		return [...presets, ...models];
	}

	private async getCatalogModels(headers: Record<string, string>): Promise<readonly OpenRouterModel[]> {
		const url = 'https://openrouter.ai/api/v1/models';
		const rsp = await this.context.fetch(url, { headers: headers });
		if (!rsp.ok) {
			throw new Error(`Getting models (${url}) failed: ${rsp.status} (${rsp.statusText})`);
		}

		type ModelsResponse = {
			data: {
				id: string;
				name: string;
				context_length: number;
				top_provider: {
					max_completion_tokens?: number;
				};
				supported_parameters?: string[];
			}[];
		};

		const results = (await rsp.json()) as ModelsResponse;
		return results.data.map<OpenRouterModel>(
			m =>
				({
					id: m.id,
					name: m.name,
					maxTokens: {
						input: m.context_length,
						output: m.top_provider?.max_completion_tokens ?? Math.floor(m.context_length / 2),
					},
					provider: provider,
					temperature: null,
					// OpenRouter hard-errors when `response_format` reaches a model that doesn't advertise support
					supportsStructuredOutputs: m.supported_parameters?.includes('structured_outputs') ?? false,
				}) satisfies OpenRouterModel,
		);
	}

	/** Surfaces the user's own OpenRouter presets as selectable models (`@preset/<slug>`), so a
	 *  named configuration kept on openrouter.ai is reachable from the model picker. The listing's
	 *  exact shape isn't documented, so only the fields we can confirm are consumed. */
	private async getPresetModels(
		headers: Record<string, string>,
		models: readonly OpenRouterModel[],
	): Promise<OpenRouterModel[]> {
		type PresetEntry = {
			id?: string;
			slug?: string;
			name?: string;
			designated_version?: { config?: { model?: string } };
		};
		type PresetsResponse = { data?: PresetEntry[] };

		try {
			const rsp = await this.context.fetch('https://openrouter.ai/api/v1/presets', { headers: headers });
			if (!rsp.ok) return [];

			const json = (await rsp.json()) as PresetsResponse | PresetEntry[];
			const entries = Array.isArray(json) ? json : (json.data ?? []);

			const presets: OpenRouterModel[] = [];
			for (const p of entries) {
				const slug = p.slug || p.id;
				if (!slug) continue;

				// The preset's pinned model, when it has one, is the only honest source for the token
				// limits; otherwise fall back to a conservative window
				const backing = models.find(m => m.id === p.designated_version?.config?.model);

				presets.push({
					id: `@preset/${slug}`,
					name: `${p.name || slug} (preset)`,
					maxTokens: backing?.maxTokens ?? { input: 128000, output: 32768 },
					provider: provider,
					temperature: null,
					// A preset can route anywhere, so nothing here advertises schema support — the base's
					// strip-and-retry learns per-preset from the first rejection
					supportsStructuredOutputs: backing?.supportsStructuredOutputs ?? false,
				} satisfies OpenRouterModel);
			}

			return presets;
		} catch {
			return [];
		}
	}

	protected override applyProviderOptions(request: ChatCompletionRequest, _model: AIModel<typeof provider.id>): void {
		const routing = this.context.getOpenRouterRouting?.();
		if (routing == null) return;

		request.provider = {
			...(routing.sort != null ? { sort: routing.sort } : undefined),
			...(routing.order?.length ? { order: routing.order } : undefined),
			...(routing.only?.length ? { only: routing.only } : undefined),
			...(routing.ignore?.length ? { ignore: routing.ignore } : undefined),
			...(routing.allowFallbacks != null ? { allow_fallbacks: routing.allowFallbacks } : undefined),
			...(routing.dataCollection != null ? { data_collection: routing.dataCollection } : undefined),
		};
	}

	protected override applyResponseFormat(
		request: ChatCompletionRequest,
		model: AIModel<typeof provider.id>,
		responseFormat: AIResponseFormat,
	): void {
		super.applyResponseFormat(request, model, responseFormat);
		// `supported_parameters` is a union across the providers serving a model — restrict routing
		// to providers that actually support every parameter sent, without dropping the user's
		// routing preferences already set by `applyProviderOptions`
		request.provider = { ...request.provider, require_parameters: true };
	}

	protected getUrl(_model: AIModel<typeof provider.id>): string {
		return 'https://openrouter.ai/api/v1/chat/completions';
	}

	protected override getHeaders<TAction extends AIActionType>(
		_action: TAction,
		apiKey: string,
		_model: AIModel<typeof provider.id>,
		_url: string,
	): Record<string, string> {
		return this.getHeadersCore(apiKey);
	}

	private getHeadersCore(apiKey: string): Record<string, string> {
		return {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			'HTTP-Referer': 'https://gitkraken.com/',
			'X-Title': 'GitKraken',
			Authorization: `Bearer ${apiKey}`,
		};
	}
}
