import { GoogleGenerativeAI } from '@google/generative-ai';
import { NextResponse } from 'next/server';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error('GEMINI_API_KEY is not set');
}

const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: 'gemini-2.0-flash-exp',
});

const codeGenerationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: 'application/json',
};

const codeSession = model.startChat({
  generationConfig: codeGenerationConfig,
  history: []
});

const CODE_GEN_PROMPT = `
You are an expert Next.js developer. Based on the user's request, create a complete Next.js application.

IMPORTANT: Return ONLY valid JSON in the following format:
{
  "projectTitle": "App Name",
  "explanation": "Brief description of what was built",
  "files": {
    "pages/index.js": {
      "code": "complete Next.js page component code here"
    },
    "components/ComponentName.js": {
      "code": "complete component code here"
    }
  },
  "generatedFiles": ["pages/index.js", "components/ComponentName.js"]
}

REQUIREMENTS:
- Use Next.js 14+ with App Router or Pages Router structure
- ALWAYS add "use client" directive at the top of components that use hooks or interactivity
- Use React with modern hooks and functional components
- Use Tailwind CSS for all styling (classes like bg-blue-500, text-white, etc.)
- Create multiple components when needed in a 'components/' folder
- Use proper imports: import React from 'react'; for all components
- Available icons from lucide-react: Heart, Shield, Clock, Users, Play, Home, Search, Menu, User, Settings, Mail, Bell, Calendar, Star, Upload, Download, Trash, Edit, Plus, Minus, Check, X, ArrowRight
- Example icon usage: import { Heart, Star } from 'lucide-react'; then <Heart className="w-5 h-5" />
- Use state management with useState and useEffect when needed
- Optional packages when relevant: date-fns, react-chartjs-2, firebase
- For images, use: https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800 or similar valid Unsplash URLs
- Add emojis for better UX where appropriate
- Make it production-ready with beautiful, modern design
- Ensure all imports work correctly (no nested folder imports)
- For Next.js pages, use: export default function PageName() { return (...); }
- For components, use: export default function ComponentName() { return (...); }

Create a fully functional, beautiful Next.js application based on the user's request.
`;

export async function POST(req: Request) {
  try {
    const { prompt } = await req.json();

    if (!prompt) {
      return NextResponse.json({ error: 'Prompt is required' }, { status: 400 });
    }

    const finalPrompt = prompt + '\n\n' + CODE_GEN_PROMPT;
    const result = await codeSession.sendMessage(finalPrompt);
    const response = result.response.text();
    
    // Parse the JSON response
    const parsedResponse = JSON.parse(response);
    
    return NextResponse.json(parsedResponse);
  } catch (error) {
    console.error('AI Code Generation Error:', error);
    return NextResponse.json(
      { error: 'Failed to generate code' },
      { status: 500 }
    );
  }
}