const state = {
    mode: "rag",
    provider: "local"
};

const $ = (id) => document.getElementById(id);

function setActive(buttonA, buttonB, activeA) {
    buttonA.classList.toggle("active", activeA);
    buttonB.classList.toggle("active", !activeA);
}

function updateUI() {
    const rag = state.mode === "rag";
    const google = state.provider === "google";

    setActive(
        $("ragMode"),
        $("chatMode"),
        rag
    );

    setActive(
        $("localProvider"),
        $("googleProvider"),
        !google
    );

    $("pdfPanel").classList.toggle("hidden", !rag);
    $("ragSettings").classList.toggle("hidden", !rag);
    $("apiKeyBox").classList.toggle("hidden", !google);

    $("title").textContent =
        rag ? "Document RAG" : "Normal Chatbot";

    $("subtitle").textContent =
        rag
            ? "Ask questions about your indexed PDF."
            : "General-purpose chat without document retrieval.";

    $("modeHelp").textContent =
        rag
            ? "Only retrieved PDF context is given to the model."
            : "The model answers normally without RAG.";

    $("question").placeholder =
        rag
            ? "Ask a question about the PDF…"
            : "Ask anything…";
}


/* -----------------------------
   Mode / Provider
----------------------------- */

$("ragMode").addEventListener("click", () => {
    state.mode = "rag";
    updateUI();
});

$("chatMode").addEventListener("click", () => {
    state.mode = "chat";
    updateUI();
});

$("localProvider").addEventListener("click", () => {
    state.provider = "local";
    updateUI();
});

$("googleProvider").addEventListener("click", () => {
    state.provider = "google";
    updateUI();
});


/* -----------------------------
   Top K
----------------------------- */

$("topK").addEventListener("input", () => {
    $("topKValue").textContent = $("topK").value;
});


/* -----------------------------
   Clear Chat
----------------------------- */

$("clearBtn").addEventListener("click", () => {
    $("chat").innerHTML = "";
    $("timingHint").textContent = "";
});


/* -----------------------------
   Safe JSON Request
----------------------------- */

async function apiRequest(url, options = {}) {

    try {

        const response = await fetch(url, options);

        const contentType =
            response.headers.get("content-type") || "";

        let data;

        if (contentType.includes("application/json")) {
            data = await response.json();
        } else {
            const text = await response.text();

            data = {
                detail: text || "Server returned an empty response."
            };
        }

        if (!response.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `HTTP ${response.status}`
            );
        }

        return data;

    } catch (error) {

        console.error("API request failed:", {
            url,
            error
        });

        throw error;
    }
}


/* -----------------------------
   PDF Upload
----------------------------- */

$("uploadBtn").addEventListener("click", async () => {

    const fileInput = $("pdfFile");
    const file = fileInput.files[0];

    if (!file) {
        $("uploadStatus").textContent =
            "Choose a PDF first.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        $("uploadStatus").textContent =
            "Only PDF files are supported.";
        return;
    }

    $("uploadBtn").disabled = true;
    $("uploadBtn").textContent = "Indexing…";

    $("uploadStatus").textContent =
        "Extracting text, creating embeddings and building HNSW index…";

    const formData = new FormData();

    formData.append("file", file);

    try {

        const data = await apiRequest(
            "/api/upload",
            {
                method: "POST",
                body: formData
            }
        );

        $("uploadStatus").textContent =
            data.message || "PDF indexed successfully.";

        $("dbPdf").textContent =
            data.filename || file.name;

        $("dbChunks").textContent =
            data.chunks ?? "—";

    } catch (error) {

        $("uploadStatus").textContent =
            `Error: ${error.message}`;

    } finally {

        $("uploadBtn").disabled = false;
        $("uploadBtn").textContent = "Index PDF";
    }
});


/* -----------------------------
   Add Message
----------------------------- */

function addMessage(
    role,
    text,
    meta = "",
    sources = []
) {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        `message ${role}`;

    const bubble =
        document.createElement("div");

    bubble.className = "bubble";

    bubble.textContent = text;

    wrapper.appendChild(bubble);


    if (meta) {

        const metaEl =
            document.createElement("div");

        metaEl.className = "meta";

        metaEl.textContent = meta;

        wrapper.appendChild(metaEl);
    }


    if (sources && sources.length > 0) {

        const sourceBox =
            document.createElement("div");

        sourceBox.className = "sources";

        sources.forEach((source) => {

            const item =
                document.createElement("div");

            item.className = "source";

            const rank =
                source.rank ?? "";

            const page =
                source.page ?? "?";

            const distance =
                source.distance ?? "";

            item.textContent =
                `[${rank}] Page ${page} · distance ${distance}`;

            sourceBox.appendChild(item);
        });

        wrapper.appendChild(sourceBox);
    }


    $("chat").appendChild(wrapper);

    $("chat").scrollTop =
        $("chat").scrollHeight;

    return wrapper;
}


/* -----------------------------
   Send Chat
----------------------------- */

async function sendMessage() {

    const question =
        $("question").value.trim();

    if (!question) {
        return;
    }


    addMessage(
        "user",
        question
    );

    $("question").value = "";

    const loading =
        addMessage(
            "assistant",
            "Thinking…"
        );

    $("sendBtn").disabled = true;


    try {

        const body = {
            question: question,
            mode: state.mode,
            provider: state.provider,
            top_k: Number($("topK").value),
            api_key:
                state.provider === "google"
                    ? $("apiKey").value.trim() || null
                    : null
        };


        console.log(
            "Sending request:",
            body
        );


        const data =
            await apiRequest(
                "/api/chat",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify(body)
                }
            );


        loading.remove();


        /* -----------------------------
           Timing
        ----------------------------- */

        let timing = "";

        if (data.timing) {

            const total =
                Number(data.timing.total_ms || 0);

            const generation =
                data.timing.generation || {};

            const generationMs =
                Number(
                    generation.generation_ms || 0
                );


            if (data.timing.retrieval) {

                const embeddingMs =
                    Number(
                        data.timing.retrieval.embedding_ms || 0
                    );

                const hnswMs =
                    Number(
                        data.timing.retrieval.hnsw_ms || 0
                    );

                timing =
                    `Total ${total.toFixed(0)} ms · ` +
                    `Embedding ${embeddingMs.toFixed(0)} ms · ` +
                    `HNSW ${hnswMs.toFixed(0)} ms · ` +
                    `Generation ${generationMs.toFixed(0)} ms`;

            } else {

                timing =
                    `Total ${total.toFixed(0)} ms · ` +
                    `Generation ${generationMs.toFixed(0)} ms`;
            }


            const tokensPerSecond =
                Number(
                    generation.tokens_per_second || 0
                );

            if (tokensPerSecond > 0) {

                timing +=
                    ` · ${tokensPerSecond.toFixed(2)} tok/s`;
            }
        }


        addMessage(
            "assistant",
            data.answer || "No answer returned.",
            timing,
            data.sources || []
        );

        $("timingHint").textContent =
            timing;


    } catch (error) {

        loading.remove();

        console.error(
            "Chat error:",
            error
        );

        addMessage(
            "assistant",
            `Error: ${error.message}`
        );

    } finally {

        $("sendBtn").disabled = false;

        $("question").focus();
    }
}


/* -----------------------------
   Chat Form
----------------------------- */

$("chatForm").addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        await sendMessage();
    }
);


/* -----------------------------
   Enter Key
----------------------------- */

$("question").addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


/* -----------------------------
   Server Status
----------------------------- */

async function loadStatus() {

    try {

        const data =
            await apiRequest(
                "/api/status"
            );


        $("statusDot")
            .classList
            .add("ok");

        $("serverStatus").textContent =
            "Server connected";


        $("dbPdf").textContent =
            data.pdf || "—";

        $("dbChunks").textContent =
            data.chunks ?? "—";


        if (!data.rag_loaded) {

            $("uploadStatus").textContent =
                "No RAG collection loaded. Upload a PDF.";
        }


    } catch (error) {

        console.error(
            "Status error:",
            error
        );

        $("serverStatus").textContent =
            "Server unavailable";

        $("uploadStatus").textContent =
            "Could not connect to FastAPI server.";
    }
}


/* -----------------------------
   Initial UI
----------------------------- */

updateUI();

loadStatus();