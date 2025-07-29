import * as z from "zod";
import { inngest } from "./client";
import { getSandbox, lastAssistantTextMessageContent } from "./utils";
import {
  openai,
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

interface AgentState {
  summary: string;
  files: Record<string, string>;
}

const modelProvider = openai({
  baseUrl: process.env.OPENROUTER_BASE_URL!,
  apiKey: process.env.OPENROUTER_API_KEY!,
  model: "deepseek/deepseek-r1-0528-qwen3-8b:free", // ✅ You can change to another DeepSeek variant
});

export const codeAgentFunction = inngest.createFunction(
  { id: "code-agent" },
  { event: "code-agent/run" },
  async ({ event, step }) => {
    const sandboxId = await step.run("get-sandbox-id", async () => {
      const sandbox = await Sandbox.create("vibegourab");
      return sandbox.sandboxId;
    });

    const previousMessages = await step.run("get-previous-messages", async () => {
      const messages = await prisma.message.findMany({
        where: { projectId: event.data.projectId },
        orderBy: { createdAt: "desc" },
        take: 5,
      });

      return messages
        .reverse()
        .map((message): Message => ({
          content: message.content,
          role: message.role === "ASSISTANT" ? "assistant" : "user",
          type: "text",
        }));
    });

    const state = createState<AgentState>(
      { summary: "", files: {} },
      { messages: previousMessages }
    );

    const codeAgent = createAgent<AgentState>({
      name: "code-agent",
      description: "An expert coding agent",
      system: PROMPT,
      model: modelProvider,
      tools: [
        createTool({
          name: "terminal",
          description: "Use the terminal to run shell commands.",
          parameters: z.object({ command: z.string() }),
          handler: async ({ command }, { step }) => {
            return await step?.run("terminal", async () => {
              const sandbox = await getSandbox(sandboxId);
              const buffers = { stdout: "", stderr: "" };
              try {
                const result = await sandbox.commands.run(command, {
                  onStdout: (d) => (buffers.stdout += d),
                  onStderr: (d) => (buffers.stderr += d),
                });
                return result.stdout;
              } catch (error) {
                return `Command failed: ${error}\nstdout: ${buffers.stdout}\nstderr: ${buffers.stderr}`;
              }
            });
          },
        }),
        createTool({
          name: "createOrUpdateFiles",
          description: "Create or update files in the Sandbox.",
          parameters: z.object({
            files: z.array(z.object({ path: z.string(), content: z.string() })),
          }),
          handler: async ({ files }, { step, network }) => {
            const newFiles = await step?.run("create-or-update-files", async () => {
              const sandbox = await getSandbox(sandboxId);
              const updatedFiles = { ...(network.state.data.files || {}) };

              for (const file of files) {
                await sandbox.files.write(file.path, file.content);
                updatedFiles[file.path] = file.content;
              }
              return updatedFiles;
            });

            if (typeof newFiles === "object") {
              network.state.data.files = newFiles;
            }
          },
        }),
        createTool({
          name: "readFiles",
          description: "Read files from the Sandbox.",
          parameters: z.object({ files: z.array(z.string()) }),
          handler: async ({ files }, { step }) => {
            return await step?.run("read-files", async () => {
              const sandbox = await getSandbox(sandboxId);
              const contents: Record<string, string>[] = [];

              for (const file of files) {
                const content = await sandbox.files.read(file);
                contents.push({ path: file, content });
              }

              return JSON.stringify(contents);
            });
          },
        }),
      ],
      lifecycle: {
        onResponse: async ({ result, network }) => {
          const summary = lastAssistantTextMessageContent(result);
          if (summary?.includes("<task_summary>")) {
            network.state.data.summary = summary;
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
        if (!network.state.data.summary) return codeAgent;
      },
    });

    const result = await network.run(event.data.value, { state });

    const fragmentTitleGenerator = createAgent({
      name: "fragment-title-generator",
      description: "Generates a short, descriptive title for a code fragment",
      system: FRAGMENT_TITLE_PROMPT,
      model: modelProvider,
    });

    const responseGenerator = createAgent({
      name: "response-generator",
      description: "Generates a user-friendly message explaining what was built",
      system: RESPONSE_PROMPT,
      model: modelProvider,
    });

    const { output: fragmentTitleOutput } = await fragmentTitleGenerator.run(
      result.state.data.summary
    );
    const { output: responseOutput } = await responseGenerator.run(result.state.data.summary);

    const generateText = (output: any, fallback: string) => {
      if (output?.[0]?.type !== "text") return fallback;
      const content = output[0].content;
      return typeof content === "string"
        ? content.trim()
        : content.map((item: any) => item.text).join(" ");
    };

    const isError =
      !result.state.data.summary || Object.keys(result.state.data.files || {}).length === 0;

    const sandboxUrl = await step.run("get-sandbox-url", async () => {
      const sandbox = await getSandbox(sandboxId);
      return `https://${sandbox.getHost(3000)}`;
    });

    await step.run("save-result", async () => {
      if (isError) {
        return prisma.message.create({
          data: {
            projectId: event.data.projectId,
            content: "Error: No summary or files generated.",
            role: "ASSISTANT",
            type: "ERROR",
          },
        });
      }

      return prisma.message.create({
        data: {
          projectId: event.data.projectId,
          content: generateText(responseOutput, "Here's what I built for you."),
          role: "ASSISTANT",
          type: "RESULT",
          fragment: {
            create: {
              sandboxUrl,
              title: generateText(fragmentTitleOutput, "Fragment"),
              files: result.state.data.files || {},
            },
          },
        },
      });
    });

    return {
      url: sandboxUrl,
      title: "Fragment",
      files: result.state.data.files,
      summary: result.state.data.summary,
    };
  }
);