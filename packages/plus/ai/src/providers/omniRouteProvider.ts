import { omniRouteProviderDescriptor as provider } from '../constants.js';
import type { AIActionType, AIModel } from '../models/model.js';
import { OpenAICompatibleProviderBase } from './openAICompatibleProviderBase.js';

type OmniRouteModel = AIModel<typeof provider.id>;

const defaultBaseUrl = 'http://localhost:20128/v1';
/** Stood in for a key when the gateway is unauthenticated — never sent as a header */
const unauthenticatedKey = '<not applicable>';

export class OmniRouteProvider extends OpenAICompatibleProviderBase<typeof provider.id> {
	readonly id = provider.id;
	readonly name = provider.name;
	readonly supportsTools = true;
	protected readonly descriptor = provider;
	// No `keyValidator`: OmniRoute's key format is its own, and the key is optional entirely
	protected readonly config = {};

	override async configured(silent: boolean): Promise<boolean> {
		const url = await this.getOrPromptBaseUrl(silent);
		if (url == null) return false;

		// A key is optional — reachability is the only thing that makes OmniRoute usable
		return this.validateUrl(url);
	}

	/** Never prompts: OmniRoute answers unauthenticated on a fresh self-hosted install, so a
	 *  blocking key prompt would gate a provider that needs no key. A key stored via the picker's
	 *  "Configure AI Key" button is used when present. */
	override async getApiKey(_silent: boolean): Promise<string | undefined> {
		const cfg = this.context.getProviderConfig(this.id);
		if (!cfg.enabled) return undefined;
		if (cfg.key) return cfg.key;

		// `true` — the silent path reads the stored secret without prompting or gating on an account
		return (await super.getApiKey(true)) ?? unauthenticatedKey;
	}

	async getModels(): Promise<readonly AIModel<typeof provider.id>[]> {
		try {
			const url = this.getBaseUrl();
			if (url == null) return [];

			const apiKey = await this.getApiKey(true);
			const rsp = await this.context.fetch(`${url}/models`, {
				headers: this.getHeadersCore(apiKey),
				method: 'GET',
			});
			if (!rsp.ok) return [];

			interface OmniRouteModelsResponse {
				data: { id: string; name?: string; context_length?: number; max_output_tokens?: number }[];
			}

			const result = (await rsp.json()) as OmniRouteModelsResponse;

			return (result.data ?? []).map<OmniRouteModel>(m => {
				const input = m.context_length ?? 128000;

				return {
					id: m.id,
					name: m.name || m.id,
					maxTokens: { input: input, output: m.max_output_tokens ?? Math.floor(input / 2) },
					provider: provider,
				} satisfies OmniRouteModel;
			});
		} catch {
			return [];
		}
	}

	protected getUrl(_model: AIModel<typeof provider.id>): string | undefined {
		const url = this.getBaseUrl();
		return url ? `${url}/chat/completions` : undefined;
	}

	protected override getHeaders<TAction extends AIActionType>(
		_action: TAction,
		apiKey: string,
		_model: AIModel<typeof provider.id>,
		_url: string,
	): Record<string, string> {
		return this.getHeadersCore(apiKey);
	}

	private getBaseUrl(): string | undefined {
		const cfg = this.context.getProviderConfig(this.id);
		if (!cfg.enabled) return undefined;

		return cfg.url || defaultBaseUrl;
	}

	private getHeadersCore(apiKey: string | undefined): Record<string, string> {
		return {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(apiKey && apiKey !== unauthenticatedKey ? { Authorization: `Bearer ${apiKey}` } : undefined),
		};
	}

	private async getOrPromptBaseUrl(silent: boolean): Promise<string | undefined> {
		const cfg = this.context.getProviderConfig(this.id);
		if (!cfg.enabled) return undefined;
		if (cfg.url) return cfg.url;

		const url = await this.context.getOrPromptUrl(
			this.id,
			{
				currentUrl: defaultBaseUrl,
				title: 'Connect to OmniRoute',
				placeholder: 'Please enter your OmniRoute gateway URL to use this feature',
				validator: async (u: string) => {
					const valid = await this.validateUrl(u);
					return valid ? undefined : 'Could not connect to the OmniRoute gateway. Make sure it is running.';
				},
			},
			silent,
		);

		return url ?? defaultBaseUrl;
	}

	private async validateUrl(url: string): Promise<boolean> {
		try {
			const apiKey = await this.getApiKey(true);
			const rsp = await this.context.fetch(`${url}/models`, {
				headers: this.getHeadersCore(apiKey),
				method: 'GET',
			});
			return rsp.ok;
		} catch {
			return false;
		}
	}
}
