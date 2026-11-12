//! One deadline for a whole request body.
//!
//! tower-http's `RequestBodyTimeoutLayer` restarts its timer on every
//! frame, so a client that sends a byte at a time never times out and can
//! hold a connection, and a transcription admission, for as long as it
//! likes. This body fails once its deadline passes, however steadily the
//! client trickles.

use std::{
    future::Future as _,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    BoxError,
    body::{Body, Bytes},
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use http_body::{Frame, SizeHint};
use tokio::time::Sleep;

/// The request body did not arrive in time.
#[derive(Debug, thiserror::Error)]
#[error("the request body did not arrive within {0:?}")]
struct TimedOut(Duration);

/// Whether `error`, or an error it wraps, is a request body that missed
/// its deadline. axum reports a failed body read as a bad request or, in
/// a multipart upload, as an internal error; this is neither.
pub(crate) fn timed_out(error: &(dyn std::error::Error + 'static)) -> bool {
    std::iter::successors(Some(error), |error| error.source())
        .any(<dyn std::error::Error>::is::<TimedOut>)
}

/// The message a request whose body missed its deadline is answered with.
pub(crate) const TIMED_OUT: &str = "the request body did not arrive in time";

/// A request body that fails once `deadline` passes.
struct DeadlineBody {
    inner: Body,
    deadline: Pin<Box<Sleep>>,
    timeout: Duration,
}

impl http_body::Body for DeadlineBody {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        let this = self.get_mut();
        // The deadline first, so a client that always has a byte ready
        // cannot outrun it.
        if this.deadline.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Some(Err(Box::new(TimedOut(this.timeout)))));
        }
        Pin::new(&mut this.inner)
            .poll_frame(cx)
            .map_err(BoxError::from)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// Middleware giving each request body `timeout` from when its headers
/// were read.
pub(crate) async fn limit(
    State(timeout): State<Duration>,
    request: Request,
    next: Next,
) -> Response {
    let request = request.map(|inner| {
        Body::new(DeadlineBody {
            inner,
            deadline: Box::pin(tokio::time::sleep(timeout)),
            timeout,
        })
    });
    next.run(request).await
}
