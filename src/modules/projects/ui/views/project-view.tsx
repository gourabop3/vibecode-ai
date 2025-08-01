"use client";
;
import Link from "next/link";
import { Suspense, useState } from "react";

import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from "@/components/ui/tabs";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from "@/components/ui/resizable";
import { MessagesContainer } from "../components/messages-container";
import { Fragment } from "@/generated/prisma";
import { Header } from "../components/header";
import { FragmentWeb } from "../components/fragment-web";
import { CodeIcon, CrownIcon, EyeIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FileExplorer } from "@/components/file-explorer";
import { UserControl } from "@/components/user-control";
import { useAuth } from "@clerk/nextjs";
import { Loader } from "@/components/ui/loader";

interface ProjectViewProps {
  projectId: string;
}

export const ProjectView = ({
  projectId
}: ProjectViewProps) => {

  const { has } = useAuth();
  const hasPremiumAccess = has?.({
    plan : "pro"
  });
  const [activeFragment, setActiveFragment] = useState<Fragment | null>(null);
  const [tabState, setTabState] = useState<"preview"|"code">("preview");

  return (
    <div className="h-full">
      <ResizablePanelGroup direction="horizontal" className="h-full">
        <ResizablePanel
          defaultSize={35}
          minSize={20}
          className="flex flex-col min-h-0"
        >
          <Suspense fallback={<Loader/>}>
            <Header projectId={projectId}/>
          </Suspense>
          <Suspense fallback={<Loader/>}>
            <MessagesContainer
              projectId={projectId}
              activeFragment={activeFragment}
              setActiveFragment={setActiveFragment}
            />
          </Suspense>
        </ResizablePanel>
        <ResizableHandle withHandle />
        <ResizablePanel
          defaultSize={65}
          minSize={50}
        >
          <Tabs
            className="h-full gap-0"
            defaultValue="preview"
            value={tabState}
            onValueChange={(value)=>setTabState(value as "preview"|"code")}
          >
            <div className="w-full flex items-center p-2 border-b gap-x-2">
              <TabsList className="h-8 p-0 border rounded-md">
                <TabsTrigger value="preview" className="rounded-md">
                  <EyeIcon/>
                </TabsTrigger>
                <TabsTrigger value="code" className="rounded-md">
                  <CodeIcon/>
                </TabsTrigger>
              </TabsList>
              <div className="ml-auto flex items-center gap-x-2">
                {
                  !hasPremiumAccess && (
                    <Button
                      asChild
                      size="sm"
                    >
                      <Link href="/pricing">
                        <CrownIcon/>
                        Upgrade
                      </Link>
                    </Button>
                  )
                }
                <UserControl/>
              </div>
            </div>
            <TabsContent value="preview">
              {
                !!activeFragment && (
                  <FragmentWeb
                    fragment={activeFragment}
                  />
                )
              }
            </TabsContent>
            <TabsContent value="code" className="min-h-0">
              {
                !!activeFragment && (
                  <FileExplorer files={activeFragment.files as { [path: string]: string }} />
                )
              }
            </TabsContent>
          </Tabs>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}

