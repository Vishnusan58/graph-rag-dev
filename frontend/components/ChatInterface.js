import { useState, useRef, useEffect } from 'react';
import styles from '../styles/ChatInterface.module.css';

const ChatInterface = ({ messages, onSendMessage, loading }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  
  // Auto-scroll to the bottom of the chat when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !loading) {
      onSendMessage(input);
      setInput('');
    }
  };
  
  // Function to render code blocks in messages
  const renderMessageContent = (content) => {
    // Split the content by code blocks
    const parts = content.split(/(```[\s\S]*?```)/g);
    
    return parts.map((part, index) => {
      // Check if this part is a code block
      if (part.startsWith('```') && part.endsWith('```')) {
        // Extract language and code
        const match = part.match(/```(?:(\w+))?\n([\s\S]*?)```/);
        if (match) {
          const [, language, code] = match;
          return (
            <div key={index} className={styles.codeBlock}>
              {language && <div className={styles.codeLanguage}>{language}</div>}
              <pre>
                <code>{code}</code>
              </pre>
            </div>
          );
        }
      }
      
      // Regular text
      return <p key={index}>{part}</p>;
    });
  };
  
  return (
    <div className={styles.chatInterface}>
      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <p>Ask me anything about programming!</p>
            <p>For example:</p>
            <ul>
              <li>"How does ownership work in Rust?"</li>
              <li>"Explain the difference between borrowing and references"</li>
              <li>"Show me an example of using iterators"</li>
            </ul>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`${styles.message} ${
                message.role === 'user' ? styles.userMessage : styles.assistantMessage
              }`}
            >
              <div className={styles.messageContent}>
                {renderMessageContent(message.content)}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.loadingIndicator}>
              <div className={styles.dot}></div>
              <div className={styles.dot}></div>
              <div className={styles.dot}></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className={styles.inputForm} onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className={styles.inputField}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className={styles.sendButton}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;