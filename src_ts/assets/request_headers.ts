import { z } from "zod";

export const CommandRequestSchema = z.object({
  command: z.string(),
  id: z.string(),
  fetch_manual: z.boolean().optional().default(true),
  server_name: z.string().optional(),
  language: z.string().optional(),
  code: z.string().optional(),
});

export type CommandRequest = z.infer<typeof CommandRequestSchema>;

export type OrchestratorResponse =
  | {
      status: number;
      result: string | null;
      info?: string;
      id: string;
    }
  | {
      status: number;
      detail: string;
      id: string;
    };

