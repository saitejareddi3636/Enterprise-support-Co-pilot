"use client";

import { FormEvent, useState } from "react";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isAsking, setIsAsking] = useState(false);

  const handleUpload = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) {
      return;
    }
    // Wire up backend upload API here later.
  };

  const handleAsk = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }

    setIsAsking(true);
    try {
      // Wire up backend question answering API here later.
      setAnswer("");
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <main className="page">
      <div className="container">
        <h1 className="title">Enterprise Support Copilot</h1>

        <section className="section">
          <h2 className="sectionTitle">Upload documents</h2>
          <form className="card" onSubmit={handleUpload}>
            <label className="label">
              <span>Select file</span>
              <input
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button type="submit" className="button" disabled={!file}>
              Upload
            </button>
          </form>
        </section>

        <section className="section">
          <h2 className="sectionTitle">Ask a question</h2>
          <form className="card" onSubmit={handleAsk}>
            <label className="label">
              <span>Your question</span>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
                placeholder="Ask about your support knowledge base..."
              />
            </label>
            <button type="submit" className="button" disabled={isAsking}>
              {isAsking ? "Thinking..." : "Ask"}
            </button>
          </form>
        </section>

        <section className="section">
          <h2 className="sectionTitle">Answer</h2>
          <div className="card">
            {answer ? (
              <p>{answer}</p>
            ) : (
              <p className="muted">Answers will appear here.</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

