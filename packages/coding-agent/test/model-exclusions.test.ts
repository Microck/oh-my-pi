import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import type { Model } from "@oh-my-pi/pi-ai";
import { parseArgs } from "@oh-my-pi/pi-coding-agent/cli/args";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { cfgEnabledModels, cfgExcludedModels } from "@oh-my-pi/pi-coding-agent/config/model-settings";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import type { ProviderModelConfig } from "@oh-my-pi/pi-coding-agent/extensibility/extensions";
import { resolveScopedModels, toSessionScopedModels, watchScopedModelSettings } from "@oh-my-pi/pi-coding-agent/main";
import { AcpAgent } from "@oh-my-pi/pi-coding-agent/modes/acp/acp-agent";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import { AgentSideConnection, type AnyMessage } from "@oh-my-pi/pi-utils/acp";

function modelDefinition(id: string): ProviderModelConfig {
	return {
		id,
		name: id,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 8_192,
	};
}

function selectors(models: readonly Model[]): string[] {
	return models.map(model => `${model.provider}/${model.id}`);
}

describe("excludedModels catalog policy", () => {
	let directory: TempDir;
	let auth: AuthStorage;
	let settings: Settings;
	let registry: ModelRegistry;
	let session: AgentSession | undefined;
	let acp: AcpAgent | undefined;

	beforeEach(async () => {
		directory = TempDir.createSync("@omp-model-exclusions-");
		auth = await AuthStorage.create(":memory:");
		settings = Settings.isolated();
		const modelsPath = path.join(directory.path(), "models.yml");
		// JSON is also valid YAML. Keep provider configuration and auth real and isolated.
		await Bun.write(
			modelsPath,
			JSON.stringify({
				providers: Object.fromEntries(
					Object.entries({
						devin: ["fusion-test", "fusion-test-v2", "regular-test", "regular-test-v2"],
						other: ["fusion-test"],
					}).map(([provider, ids]) => [
						provider,
						{
							baseUrl: "https://example.invalid/v1",
							api: "openai-completions",
							apiKey: "fixture-key",
							models: ids.map(modelDefinition),
						},
					]),
				),
			}),
		);
		registry = new ModelRegistry(auth, modelsPath, { settings });
	});

	afterEach(async () => {
		await acp?.dispose();
		acp = undefined;
		await session?.dispose();
		session = undefined;
		auth.close();
		await directory.remove();
	});

	function startSession(): AgentSession {
		const model = registry.find("devin", "regular-test");
		if (!model) throw new Error("Missing regular Devin fixture");
		session = new AgentSession({
			agent: new Agent({ initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] } }),
			sessionManager: SessionManager.create(directory.path(), path.join(directory.path(), "sessions")),
			settings,
			modelRegistry: registry,
			scopedModels: registry
				.getAvailableForProviders(new Set(["devin"]))
				.filter(model => ["regular-test", "regular-test-v2", "fusion-test"].includes(model.id))
				.map(model => ({ model })),
		});
		return session;
	}

	it("hides an exact provider-qualified ID from lazy lookup and the full catalog without hiding another provider's ID", () => {
		cfgExcludedModels.set(settings, ["devin/fusion-test"]);
		expect(registry.find("devin", "fusion-test")).toBeUndefined();
		expect(selectors(registry.getAvailableForProviders(new Set(["devin"])))).not.toContain("devin/fusion-test");
		expect(selectors(registry.getAll("all"))).not.toContain("devin/fusion-test");
		expect(registry.find("other", "fusion-test")?.id).toBe("fusion-test");
		expect(registry.find("devin", "fusion-test-v2")?.id).toBe("fusion-test-v2");
		expect(selectors(registry.getAvailable())).not.toContain("devin/fusion-test");
	});

	it("excludes a copied bundled selector containing glob characters", () => {
		const provider = "zhipu-coding-plan";
		const id = "glm-5.2-highspeed[1m]";
		const selector = `${provider}/${id}`;
		auth.keys.setRuntime(provider, "fixture-key");
		expect(registry.find(provider, id)).toBeDefined();
		expect(selectors(registry.getAvailableForProviders(new Set([provider])))).toContain(selector);

		cfgExcludedModels.set(settings, [selector.toUpperCase()]);
		expect(registry.find(provider, id)).toBeUndefined();
		expect(selectors(registry.getAll())).not.toContain(selector);
		expect(selectors(registry.getAvailable())).not.toContain(selector);
		expect(registry.find(provider, "glm-5.2-highspeed")).toBeDefined();

		cfgExcludedModels.set(settings, []);
		expect(registry.find(provider, id)).toBeDefined();
		expect(selectors(registry.getAvailable())).toContain(selector);
	});

	it("omits devin/fusion-* from the normal catalog and ACP options while preserving regular Devin models and auth", async () => {
		cfgExcludedModels.set(settings, ["devin/fusion-*"]);
		const live = startSession();
		const catalog = selectors(live.getAvailableModels());
		expect(catalog.filter(selector => selector.startsWith("devin/fusion-"))).toEqual([]);
		expect(catalog).toContain("devin/regular-test");
		expect(catalog).toContain("other/fusion-test");
		expect(selectors(registry.getProviderModels("devin"))).not.toContain("devin/fusion-test-v2");
		expect(registry.getProviderBaseUrl("devin")).toBe("https://example.invalid/v1");
		expect(await registry.getApiKeyForProvider("devin")).toBe("fixture-key");

		// Exercise the real ACP connection with an in-memory client transport.
		const input = new TransformStream<AnyMessage>();
		const writer = input.writable.getWriter();
		new AgentSideConnection(
			connection => {
				const agent = new AcpAgent(connection, async () => live);
				acp = agent;
				return agent;
			},
			{ readable: input.readable, writable: new WritableStream<AnyMessage>() },
		);
		if (!acp) throw new Error("ACP connection did not create an agent");
		const response = await acp.newSession({ cwd: directory.path(), mcpServers: [] });
		const modelOption = response.configOptions?.find(option => option.id === "model");
		if (!modelOption || modelOption.type !== "select") throw new Error("ACP did not advertise model options");
		const advertised = modelOption.options.flatMap(option => ("value" in option ? [option.value] : []));
		expect(advertised).toEqual(catalog);
		expect(advertised.filter(selector => selector.startsWith("devin/fusion-"))).toEqual([]);
		expect(advertised).toContain("devin/regular-test");
		await writer.close();
	});

	it("keeps exclusions after refresh and filters newly registered or discovered models", async () => {
		cfgExcludedModels.set(settings, ["devin/fusion-*"]);
		registry.registerProvider("devin", {
			baseUrl: "https://example.invalid/v1",
			api: "openai-completions",
			apiKey: "fixture-key",
			models: [modelDefinition("fusion-extension"), modelDefinition("regular-extension")],
			fetchDynamicModels: async () => [modelDefinition("fusion-discovered"), modelDefinition("regular-discovered")],
		});
		await registry.refreshRuntimeProviders("online");
		await registry.refresh("offline");
		const available = selectors(registry.getAvailable());
		expect(available.filter(selector => selector.startsWith("devin/fusion-"))).toEqual([]);
		expect(available).toContain("devin/regular-discovered");
	});

	it("applies live exclusions to cached catalog reads and cycling, and restores models when cleared", async () => {
		const live = startSession();
		const original = selectors(registry.getAll());
		cfgExcludedModels.set(settings, ["DEVIN/FUSION-*"]);
		expect(selectors(registry.getAll())).not.toContain("devin/fusion-test");
		expect(selectors(live.scopedModels.map(entry => entry.model))).toEqual([
			"devin/regular-test",
			"devin/regular-test-v2",
		]);
		expect((await live.cycleModel())?.model.id).toBe("regular-test-v2");
		expect((await live.cycleModel())?.model.id).toBe("regular-test");
		cfgExcludedModels.set(settings, []);
		expect(selectors(registry.getAll())).toEqual(original);
		expect(selectors(live.scopedModels.map(entry => entry.model))).toContain("devin/fusion-test");
	});

	it.each(["enabledModels", "--models"])("restores a scope built with exclusions enabled (%s)", async source => {
		const explicit = source === "--models";
		const patterns = ["devin/regular-test*", "devin/fusion-*"];
		cfgEnabledModels.set(settings, patterns);
		cfgExcludedModels.set(settings, ["devin/fusion-*"]);
		const parsed = parseArgs(explicit ? ["--models", patterns.join(",")] : []);
		const initialScope = await resolveScopedModels(parsed, registry, settings);
		const live = startSession();
		live.setScopedModels(toSessionScopedModels(initialScope, settings));
		watchScopedModelSettings(live, parsed, registry, settings);
		expect(selectors(live.scopedModels.map(entry => entry.model))).toEqual([
			"devin/regular-test",
			"devin/regular-test-v2",
		]);

		cfgExcludedModels.set(settings, []);
		// Settings listeners coalesce in a microtask; let async scope resolution settle.
		await Bun.sleep(0);
		expect(selectors(live.scopedModels.map(entry => entry.model))).toEqual([
			"devin/regular-test",
			"devin/regular-test-v2",
			"devin/fusion-test",
			"devin/fusion-test-v2",
		]);
		expect((await live.cycleModel())?.model.id).toBe("regular-test-v2");
		expect((await live.cycleModel())?.model.id).toBe("fusion-test");

		cfgExcludedModels.set(settings, ["devin/fusion-*"]);
		await Bun.sleep(0);
		expect(selectors(live.scopedModels.map(entry => entry.model))).not.toContain("devin/fusion-test");
		// An enabledModels edit must still leave an explicit CLI scope pinned.
		cfgEnabledModels.set(settings, ["devin/regular-test"]);
		await Bun.sleep(0);
		cfgExcludedModels.set(settings, []);
		await Bun.sleep(0);
		expect(live.scopedModels).toHaveLength(explicit ? 4 : 1);
	});

	it("retains the complete catalog and available model list when exclusions are omitted or empty", () => {
		const catalog = selectors(registry.getAll("all"));
		const available = selectors(registry.getAvailable());
		expect(available).toContain("devin/fusion-test");
		expect(available).toContain("devin/regular-test");
		cfgExcludedModels.set(settings, []);
		expect(selectors(registry.getAll("all"))).toEqual(catalog);
		expect(selectors(registry.getAvailable())).toEqual(available);
	});

	it("rejects malformed exclusion entries at the settings boundary", () => {
		expect(() => cfgExcludedModels.set(settings, ["fusion-*"])).toThrow("provider/id");
		expect(() => Settings.isolated({ excludedModels: [42] })).toThrow("provider/id");
		expect(selectors(registry.getAvailable())).toContain("devin/fusion-test");
	});
	it.each([
		"devin/fusion-[",
		"devin/fusion-[]",
		"devin/fusion-[!]",
		"devin/fusion-[z-a]",
		"devin/fusion-[Z-a]",
		"devin/fusion-{test,v2",
		"devin/fusion-test}",
		"devin/fusion-\\",
	])("rejects malformed glob %s at the settings boundary", pattern => {
		expect(() => cfgExcludedModels.set(settings, [pattern])).toThrow("Invalid excludedModels glob pattern");
		expect(() => Settings.isolated({ excludedModels: [pattern] })).toThrow("Invalid excludedModels glob pattern");
		expect(selectors(registry.getAvailable())).toContain("devin/fusion-test");
	});

	it.each([
		"devin/fusion-[tv]*",
		"devin/fusion-{test,test-v2}",
		"devin/{fusion-{test,test-v2},regular-test}",
		"devin/fusion-\\[test",
	])("accepts complete glob %s", pattern => {
		expect(() => cfgExcludedModels.set(settings, [pattern])).not.toThrow();
	});
});
