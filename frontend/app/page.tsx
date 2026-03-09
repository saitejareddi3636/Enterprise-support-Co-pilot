"use client";

import { FormEvent, useState } from "react";

type RetrievedChunk = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  index: number;
  heading?: string | null;
  content: string;
  score: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<RetrievedChunk[]>([]);
  const [supported, setSupported] = useState<boolean | null>(null);
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
      setAnswer("");
      setCitations([]);
      setSupported(null);

      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: question }),
      });

      if (!response.ok) {
        setAnswer("There was an error while asking the question.");
        return;
      }

      const data = await response.json();
      setAnswer(data.answer ?? "");
      setCitations(Array.isArray(data.chunks) ? data.chunks : []);
      if (typeof data.supported === "boolean") {
        setSupported(data.supported);
      } else {
        setSupported(true);
      }
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
              <div className="answer">
                {supported === false && (
                  <p className="muted">
                    This answer could not be verified from the available
                    documents. Showing the closest matches below.
                  </p>
                )}
                <p>{answer}</p>
                {citations.length > 0 && (
                  <div className="citations">
                    <h3 className="sectionTitle">Citations</h3>
                    <ul className="citationList">
                      {citations.map((chunk) => (
                        <li key={chunk.chunk_id} className="citationItem">
                          <div className="citationTitle">
                            {chunk.document_title}{" "}
                            <span className="muted">
                              (chunk {chunk.index}, score{" "}
                              {chunk.score.toFixed(3)})
                            </span>
                          </div>
                          {chunk.heading && (
                            <div className="citationHeading">
                              {chunk.heading}
                            </div>
                          )}
                          <div className="citationContent">
                            {chunk.content.slice(0, 280)}
                            {chunk.content.length > 280 ? "…" : ""}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">Answers will appear here.</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

