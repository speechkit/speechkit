//! A write timeout for connections.
//!
//! hyper has none: a response to a client that stops reading waits for
//! the socket to drain for as long as the client keeps it open, so the
//! connection stays open that long. The request's session slot is held
//! until its budget runs out, which is at least 120 s when no request
//! timeout is set. This wrapper fails a write that has made no progress
//! for the timeout, which makes hyper close the connection and drop the
//! response, freeing both sooner.

use std::{
    future::Future as _,
    io,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    time::Sleep,
};

/// `inner`, whose writes fail once one has been pending for `timeout`.
pub(crate) struct WriteTimeout<T> {
    inner: T,
    timeout: Duration,
    /// Started when a write first returns pending, cleared on progress.
    stalled: Option<Pin<Box<Sleep>>>,
}

impl<T> WriteTimeout<T> {
    pub(crate) fn new(inner: T, timeout: Duration) -> Self {
        Self {
            inner,
            timeout,
            stalled: None,
        }
    }

    /// Passes on a ready result, and turns a pending one into an error
    /// once writing has been pending for the timeout.
    fn watch<R>(
        &mut self,
        result: Poll<io::Result<R>>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<R>> {
        if result.is_ready() {
            self.stalled = None;
            return result;
        }
        let timeout = self.timeout;
        let stalled = self
            .stalled
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(timeout)));
        if stalled.as_mut().poll(cx).is_ready() {
            tracing::debug!(
                ?timeout,
                "the client stopped reading; closing the connection"
            );
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("the client took nothing for {timeout:?}"),
            )));
        }
        Poll::Pending
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for WriteTimeout<T> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for WriteTimeout<T> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_write(cx, buf);
        this.watch(result, cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_write_vectored(cx, bufs);
        this.watch(result, cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_flush(cx);
        this.watch(result, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_shutdown(cx);
        this.watch(result, cx)
    }
}
