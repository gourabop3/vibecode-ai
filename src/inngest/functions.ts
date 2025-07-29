import * as z from "zod";
import { inngest } from "./client";
import { getSandbox, lastAssistantTextMessageContent } from "./utils";
import {
    createAgent,
    createTool,
    createNetwork,
    Tool,
    Message,
    createState,
} from "@inngest/agent-kit";
import { Sandbox } from "@e2b/code-interpreter";
import { FRAGMENT_TITLE_PROMPT, PROMPT, RESPONSE_PROMPT } from "@/lib/prompt";
import { prisma } from "@/lib/db";
import OpenAI from "openai";

// Configure OpenRouter client
const openRouter = ({ model }: { model: string }) => ({
    client: new OpenAI({
        apiKey: process.env.NEXT_PUBLIC_OPENROUTER_API_KEY,
        baseURL: "https://openrouter.ai/api/v1",
    }),
    model,
});

interface AgentState {
    summary: string;
    files: Record<string, string>;
}

export const codeAgentFunction = inngest.createFunction(
    { id: "code-agent" },
    { event: "code-agent/run" },
    async ({ event, step }) => {
        let sandbox: Sandbox | null = null;
        try {
            const sandboxId = await step.run("get-sandbox-id", async () => {
                sandbox = await Sandbox.create("vibegourab");
                return sandbox.sandboxId;
            });

            const previousMessages = await step.run("get-previous-messages", async () => {
                const formattedMessages: Message[] = [];
                const messages = await prisma.message.findMany({
                    where: {
                        projectId: event.data.projectId,
                    },
                    orderBy: {
                        createdAt: "desc",
                    },
                    take: 5,
                });

                for (const message of messages) {
                    formattedMessages.push({
                        content: message.content,
                        role: message.role === "ASSISTANT" ? "assistant" : "user",
                        type: "text",
                    });
                }
                return [...formattedMessages].reverse(); // Non-destructive reverse
            });

            const state = createState<AgentState>(
                {
                    summary: "",
                    files: {},
                },
                {
                    messages: previousMessages,
                }
            );

            const codeAgent = createAgent<AgentState>({
                name: "code-agent",
                description: "An expert coding agent for Next.js development",
                system: PROMPT,
                model: openRouter({
                    model: "deepseek/deepseek-r1-distill-llama-70b:free",
                }),
                tools: [
                    createTool({
                        name: "terminal",
                        description: "Use the terminal to run shell commands.",
                        parameters: z.object({
                            command: z.string(),
                        }),
                        handler: async ({ command }, { step }) => {
                            return await step?.run("terminal", async () => {
                                const buffers = { stdout: "", stderr: "" };
                                try {
                                    if (!sandbox) throw new Error("Sandbox not initialized");
                                    const result = await sandbox.commands.run(command, {
                                        onStdout: (data) => {
                                            buffers.stdout += data;
                                        },
                                        onStderr: (data) => {
                                            buffers.stderr += data;
                                        },
                                    });
                                    return result.stdout;
                                } catch (error) {
                                    const errorMsg = `Command failed: ${error}\nstdout: ${buffers.stdout}\nstderr: ${buffers.stderr}`;
                                    console.error(errorMsg);
                                    return errorMsg;
                                }
                            });
                        },
                    }),
                    createTool({
                        name: "createOrUpdateFiles",
                        description: "Create or update files in the Sandbox.",
                        parameters: z.object({
                            files: z.array(
                                z.object({
                                    path: z.string(),
                                    content: z.string(),
                                })
                            ),
                        }),
                        handler: async ({ files }, { step, network }: Tool.Options<AgentState>) => {
                            const newFiles = await step?.run("create-or-update-files", async () => {
                                try {
                                    if (!sandbox) throw new Error("Sandbox not initialized");
                                    const updatedFiles = { ...network.state.data.files };
                                    for (const file of files) {
                                        await sandbox.files.write(file.path, file.content);
                                        updatedFiles[file.path] = file.content;
                                    }
                                    return updatedFiles;
                                } catch (error) {
                                    const errorMsg = `Error creating or updating files: ${error}`;
                                    console.error(errorMsg);
                                    throw new Error(errorMsg);
                                }
                            });

                            if (newFiles && typeof newFiles === "object" && !("message" in newFiles)) {
                                network.state.data.files = newFiles;
                                return "Files updated successfully";
                            }
                            throw new Error("Failed to update files");
                        },
                    }),
                    createTool({
                        name: "readFiles",
                        description: "Read files from the Sandbox.",
                        parameters: z.object({
                            files: z.array(z.string()),
                        }),
                        handler: async ({ files }, { step }) => {
                            return await step?.run("read-files", async () => {
                                try {
                                    if (!sandbox) throw new Error("Sandbox not initialized");
                                    const contents: Record<string, string>[] = [];
                                    for (const file of files) {
                                        const content = await sandbox.files.read(file);
                                        contents.push({ path: file, content });
                                    }
                                    return JSON.stringify(contents);
                                } catch (error) {
                                    const errorMsg = `Error reading files: ${error}`;
                                    console.error(errorMsg);
                                    return errorMsg;
                                }
                            });
                        },
                    }),
                ],
                lifecycle: {
                    onResponse: async ({ result, network }) => {
                        const lastAssistantMessageText = lastAssistantTextMessageContent(result);
                        if (lastAssistantMessageText && network) {
                            if (lastAssistantMessageText.includes("<task_summary>")) {
                                network.state.data.summary = lastAssistantMessageText;
                            }
                        }
                        return result;
                    },
                },
            });

            const network = createNetwork<AgentState>({
                name: "coding-agent-network",
                agents: [codeAgent],
                maxIter: 15,
                defaultState: state,
                router: async ({ network }) => {
                    const summary = network.state.data.summary;
                    if (summary) {
                        return;
                    }
                    return codeAgent;
                },
            });

            const result = await network.run(event.data.value, { state });

            const fragmentTitleGenerator = createAgent({
                name: "fragment-title-generator",
                description: "Generates a short, descriptive title for a code fragment",
                system: FRAGMENT_TITLE_PROMPT,
                model: openRouter({
                    model: "deepseek/deepseek-r1-distill-llama-70b:free",
                }),
            });

            const responseGenerator = createAgent({
                name: "response-generator",
                description: "Generates a user-friendly message explaining what was built",
                system: RESPONSE_PROMPT,
                model: openRouter({
                    model: "deepseek/deepseek-r1-distill-llama-70b:free",
                }),
            });

            const { output: fragmentTitleOutput } = await fragmentTitleGenerator.run(result.state.data.summary);
            const { output: responseOutput } = await responseGenerator.run(result.state.data.summary);

            const generateFragmentTitle = () => {
                if (!fragmentTitleOutput[0] || fragmentTitleOutput[0].type !== "text") {
                    return "Fragment";
                }
                const content = fragmentTitleOutput[0].content;
                if (Array.isArray(content)) {
                    return content.map((item) => item.text || "").join(" ").trim();
                }
                return typeof content === "string" ? content.trim() : "Fragment";
            };

            const generateResponse = () => {
                if (!responseOutput[0] || responseOutput[0].type !== "text") {
                    return "Here's what I built for you.";
                }
                const content = responseOutput[0].content;
                if (Array.isArray(content)) {
                    return content.map((item) => item.text || "").join(" ").trim();
                }
                return typeof content === "string" ? content.trim() : "Here's what I built for you.";
            };

            const isError = !result.state.data.summary || Object.keys(result.state.data.files || {}).length === 0;

            const sandboxUrl = await step.run("get-sandbox-url", async () => {
                if (!sandbox) throw new Error("Sandbox not initialized");
                const host = sandbox.getHost(3000);
                return `https://${host}`;
            });

            await step.run("save-result", async () => {
                if (isError) {
                    return await prisma.message.create({
                        data: {
                            projectId: event.data.projectId,
                            content: "Error: No summary or files generated.",
                            role: "ASSISTANT",
                            type: "ERROR",
                        },
                    });
                }

                await prisma.message.create({
                    data: {
                        projectId: event.data.projectId,
                        content: generateResponse(),
                        role: "ASSISTANT",
                        type: "RESULT",
                        fragment: {
                            create: {
                                sandboxUrl,
                                title: generateFragmentTitle(),
                                files: result.state.data.files || {},
                            },
                        },
                    },
                });
            });

            return {
                url: sandboxUrl,
                title: generateFragmentTitle(),
                files: result.state.data.files,
                summary: result.state.data.summary,
            };
        } finally {
            // Close sandbox to prevent resource leaks
            if (sandbox) {
                await step.run("close-sandbox", async () => {
                    await sandbox.close();
                });
            }
        }
    }
);