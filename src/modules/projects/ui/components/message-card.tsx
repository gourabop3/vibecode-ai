import Image from "next/image";
import { format } from "date-fns";

import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { Fragment, MessageRole, MessageType } from "@/generated/prisma";
import { ChevronRightIcon, Code2Icon } from "lucide-react";

interface Props {
  content: string;
  role : MessageRole;
  fragment: Fragment|null;
  createdAt: Date;
  isActive: boolean;
  onFragmentClick: (fragment: Fragment) => void;
  type: MessageType;
}

export const MessageCard = ({
  content,
  role,
  fragment,
  createdAt,
  isActive,
  onFragmentClick,
  type
}: Props) => {
  if (role === "ASSISTANT") {
    return (
      <AssistantMessage
        content={content}
        fragment={fragment}
        createdAt={createdAt}
        isActive={isActive}
        onFragmentClick={onFragmentClick}
        type={type}
      />
    );
  } 
  return (
    <UserMessage content={content}/>
  )
}


interface UserMessageProps {  content: string; }

const UserMessage = ({ content } : UserMessageProps ) => {
  return (
    <div className="flex justify-end pb-4 pr-2 pl-10">
      <Card className="rounded-lg bg-muted p-3 shadow-none border-none max-w-[80%] break-words">
        {content}
      </Card>
    </div>
  )
}

interface AssistantMessageProps {
  content: string;
  fragment: Fragment | null;
  createdAt: Date;
  isActive: boolean;
  onFragmentClick: (fragment: Fragment) => void;
  type: MessageType;
}

const AssistantMessage = ({
  content,
  fragment,
  createdAt,
  isActive,
  onFragmentClick,
  type
}: AssistantMessageProps) => {
  return (
    <div className={cn(
      "flex flex-col group px-2 pb-4",
      type === "ERROR" && "text-red-700 dark:text-red-400"
    )}>
      <div className="flex items-center gap-2 pl-2 mb-2">
        <Image
          src="/logo.svg"
          alt="Vibe"
          height={18}
          width={18}
          className="shrink-0"
        />
        <span className="text-sm font-medium">Vibe</span>
        <span className="text-sm text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">{format(createdAt, "HH:mm 'on' MMM dd, yyyy")}</span>
      </div>
      <div className="pl-8.5 flex flex-col gap-y-4">
        <span>{content}</span>
        { fragment && type==="RESULT" && (
          <FragmentCard
            fragment={fragment}
            isActive={isActive}
            onFragmentClick={() => onFragmentClick(fragment)}
          />  
        )}
      </div>
    </div>
  )
}


interface FragmentCardProps {
  fragment: Fragment;
  isActive: boolean;
  onFragmentClick: () => void;
}

const FragmentCard = ({
  fragment,
  isActive,
  onFragmentClick
} : FragmentCardProps) => {
  return (
    <button 
      className={cn(
        "flex items-start text-start gap-2 border rounded-lg bg-muted w-fit p-3 hover:bg-secondary transition-colors md:cursor-pointer",
        isActive && "bg-primary text-primary-foreground border-primary hover:bg-primary"
      )}
      onClick={onFragmentClick}
    >
      <Code2Icon className="size-4 mt-0.5"/>
      <div className="flex flex-col flex-1">
        <span className="text-sm font-medium line-clamp-1">
          {fragment.title}
        </span>
        <span className="text-sm">Preview</span>
      </div>
      <div className="flex items-center justify-center mt-0.5">
        <ChevronRightIcon className="size-4" />
      </div>
    </button>
  )
}
