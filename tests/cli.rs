use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

fn decide(input: &str, jsonl: bool) -> Output {
    // An absent checkpoint makes accidental model loading observable without a GPU.
    let root = tempfile::tempdir().unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_clef"));
    command.arg("--model-dir").arg(root.path()).arg("decide");
    if jsonl {
        command.arg("--jsonl");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn empty_jsonl_does_not_load_model() {
    let output = decide("\n  \n", true);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
}

#[test]
fn invalid_requests_fail_before_model_loading() {
    for jsonl in [false, true] {
        let output = decide(r#"{"model":"clef","state":"test","questions":{}}"#, jsonl);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("at least one question"));
    }
    let output = decide("\nnot json\n", true);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("request on line 2"));
}
