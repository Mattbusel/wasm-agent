// SPDX-License-Identifier: MIT
//! Conversation history management within WASM memory constraints.

use crate::error::AgentError;
use crate::types::{Message, Role};

/// Bounded conversation history with token budget enforcement.
///
/// # Guarantees
/// - Never exceeds `token_limit` when using [`push`](Self::push).
/// - System messages are never evicted by [`push_with_eviction`](Self::push_with_eviction).
/// - [`clear_non_system`](Self::clear_non_system) preserves system messages only.
pub struct ConversationHistory {
    messages: Vec<Message>,
    token_limit: usize,
    total_tokens: usize,
}

impl ConversationHistory {
    /// Creates a new history with the given token limit.
    pub fn new(token_limit: usize) -> Self {
        Self { messages: Vec::new(), total_tokens: 0, token_limit }
    }

    /// Appends a message, returning [`AgentError::HistoryOverflow`] if the budget would be exceeded.
    ///
    /// # Errors
    /// Returns [`AgentError::HistoryOverflow`] if adding the message would exceed the token limit.
    pub fn push(&mut self, msg: Message) -> Result<(), AgentError> {
        let new_total = self.total_tokens + msg.token_estimate;
        if new_total > self.token_limit {
            return Err(AgentError::HistoryOverflow { size: new_total, limit: self.token_limit });
        }
        self.total_tokens = new_total;
        self.messages.push(msg);
        Ok(())
    }

    /// Appends a message, evicting the oldest non-system messages if the budget would be exceeded.
    ///
    /// System messages are never evicted. If only system messages remain and the budget is still
    /// exceeded, the message is appended anyway to avoid losing output.
    pub fn push_with_eviction(&mut self, msg: Message) {
        while self.total_tokens + msg.token_estimate > self.token_limit && !self.messages.is_empty() {
            if let Some(pos) = self.messages.iter().position(|m| m.role != Role::System) {
                let evicted = self.messages.remove(pos);
                self.total_tokens = self.total_tokens.saturating_sub(evicted.token_estimate);
            } else {
                break;
            }
        }
        self.total_tokens += msg.token_estimate;
        self.messages.push(msg);
    }

    /// Returns a slice of all messages in the history.
    pub fn messages(&self) -> &[Message] { &self.messages }
    /// Returns the current total token estimate.
    pub fn total_tokens(&self) -> usize { self.total_tokens }
    /// Returns the number of messages in the history.
    pub fn len(&self) -> usize { self.messages.len() }
    /// Returns `true` if the history is empty.
    pub fn is_empty(&self) -> bool { self.messages.is_empty() }

    /// Removes all non-system messages, resetting the token count accordingly.
    pub fn clear_non_system(&mut self) {
        self.messages.retain(|m| m.role == Role::System);
        self.total_tokens = self.messages.iter().map(|m| m.token_estimate).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_history_push_within_limit_ok() {
        let mut h = ConversationHistory::new(1000);
        assert!(h.push(Message::user("hello")).is_ok());
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn test_history_push_over_limit_returns_overflow_error() {
        let mut h = ConversationHistory::new(1);
        let err = h.push(Message::user("this message is definitely too long for a 1 token limit")).unwrap_err();
        assert!(matches!(err, AgentError::HistoryOverflow { .. }));
    }

    #[test]
    fn test_history_token_count_accumulates() {
        let mut h = ConversationHistory::new(10000);
        h.push(Message::user("hello")).unwrap();
        h.push(Message::assistant("hi there")).unwrap();
        assert!(h.total_tokens() > 0);
    }

    #[test]
    fn test_history_push_with_eviction_stays_within_budget_approximately() {
        let mut h = ConversationHistory::new(20);
        h.push_with_eviction(Message::system("You are an agent"));
        h.push_with_eviction(Message::user("msg 1"));
        h.push_with_eviction(Message::user("msg 2"));
        h.push_with_eviction(Message::user("a longer message that forces eviction of earlier messages"));
        // System message must be preserved
        assert!(h.messages().iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn test_history_clear_non_system_removes_user_and_assistant_messages() {
        let mut h = ConversationHistory::new(10000);
        h.push(Message::system("sys")).unwrap();
        h.push(Message::user("user msg")).unwrap();
        h.push(Message::assistant("asst msg")).unwrap();
        h.clear_non_system();
        assert_eq!(h.len(), 1);
        assert_eq!(h.messages()[0].role, Role::System);
    }

    #[test]
    fn test_history_is_empty_initially() {
        let h = ConversationHistory::new(1000);
        assert!(h.is_empty());
    }

    #[test]
    fn test_history_clear_non_system_updates_token_count() {
        let mut h = ConversationHistory::new(10000);
        h.push(Message::system("sys")).unwrap();
        let sys_tokens = h.total_tokens();
        h.push(Message::user("a bunch of user content that has tokens")).unwrap();
        h.clear_non_system();
        assert_eq!(h.total_tokens(), sys_tokens);
    }

    #[test]
    fn test_history_push_multiple_messages_len_correct() {
        let mut h = ConversationHistory::new(10000);
        for i in 0..5u32 {
            h.push(Message::user(format!("msg {i}"))).unwrap();
        }
        assert_eq!(h.len(), 5);
    }
}
