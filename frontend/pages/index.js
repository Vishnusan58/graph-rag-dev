import { useState, useEffect } from 'react';
import Head from 'next/head';
import ChatInterface from '../components/ChatInterface';
import CodePreview from '../components/CodePreview';
import GraphView from '../components/GraphView';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [codeSnippets, setCodeSnippets] = useState([]);
  const [language, setLanguage] = useState('Rust');

  // Function to handle sending a message
  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    // Add user message to the chat
    const userMessage = { role: 'user', content: message };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setLoading(true);

    try {
      // Call the API to get a response
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          language: language,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from the server');
      }

      const data = await response.json();

      // Add assistant message to the chat
      const assistantMessage = { role: 'assistant', content: data.answer };
      setMessages((prevMessages) => [...prevMessages, assistantMessage]);

      // Extract code snippets from the response
      const extractedSnippets = extractCodeSnippets(data.answer);
      setCodeSnippets(extractedSnippets);

      // Set graph data if available
      if (data.graph_data) {
        setGraphData(data.graph_data);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message to the chat
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  // Function to extract code snippets from text
  const extractCodeSnippets = (text) => {
    const regex = /```(?:(\w+))?\n([\s\S]*?)```/g;
    const snippets = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
      snippets.push({
        language: match[1] || 'text',
        code: match[2],
      });
    }

    return snippets;
  };

  // Function to toggle graph view
  const toggleGraphView = () => {
    setShowGraph(!showGraph);
  };

  // Function to handle language change
  const handleLanguageChange = (newLanguage) => {
    setLanguage(newLanguage);
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>GraphRAG Assistant</title>
        <meta name="description" content="AI-powered programming assistant with graph-based retrieval" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>GraphRAG Assistant</h1>
        
        <div className={styles.languageSelector}>
          <label htmlFor="language-select">Programming Language: </label>
          <select
            id="language-select"
            value={language}
            onChange={(e) => handleLanguageChange(e.target.value)}
          >
            <option value="Rust">Rust</option>
            <option value="Python">Python</option>
            <option value="JavaScript">JavaScript</option>
            <option value="Go">Go</option>
          </select>
        </div>

        <div className={styles.toggleButton}>
          <button onClick={toggleGraphView}>
            {showGraph ? 'Hide Graph View' : 'Show Graph View'}
          </button>
        </div>

        <div className={styles.contentContainer}>
          <div className={styles.chatContainer}>
            <ChatInterface
              messages={messages}
              onSendMessage={handleSendMessage}
              loading={loading}
            />
          </div>

          <div className={styles.sidePanel}>
            {showGraph && graphData ? (
              <GraphView graphData={graphData} />
            ) : (
              <CodePreview snippets={codeSnippets} />
            )}
          </div>
        </div>
      </main>

      <footer className={styles.footer}>
        <p>GraphRAG Assistant - Powered by LangChain and Neo4j</p>
      </footer>
    </div>
  );
}

