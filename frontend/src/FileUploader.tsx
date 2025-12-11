import React, {useState, useCallback, type ChangeEvent, type FormEvent, useEffect} from 'react';

interface UploadResponse {
    uploaded_file: string; // Путь или имя загруженного файла
    result: string
}

const ImageUploader: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [responseJson, setResponseJson] = useState<UploadResponse | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        return () => {
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    const backendUrl = 'http://127.0.0.1:8000/upload';

    const handleFileChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        if (event.target.files && event.target.files.length > 0) {
            setSelectedFile(event.target.files[0]);
            setResponseJson(null);
            setError(null);
            const url = URL.createObjectURL(event.target.files[0]);
            setPreviewUrl(url);
        } else {
            setSelectedFile(null);
            setPreviewUrl(null);
        }
    }, [previewUrl]);

    const handleSubmit = useCallback(async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (!selectedFile) {
            setError('Please, upload a file.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResponseJson(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error: ${response.status}. Server response: ${errorText}`);
            }

            const jsonResponse: UploadResponse = await response.json();
            setResponseJson(jsonResponse);
            setSelectedFile(null);

        } catch (err) {
            if (err instanceof Error) {
                setError(`Upload error: ${err.message}`);
            } else {
                setError('Unknown error');
            }
            setResponseJson(null);
        } finally {
            setIsLoading(false);
        }
    }, [selectedFile]);

    return (
        <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', maxWidth: '1000px', margin: 'auto' }}>
            <h2>Bag of features</h2>

            <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    disabled={isLoading}
                    style={{ display: 'block', marginBottom: '10px' }}
                />

                <button
                    type="submit"
                    disabled={isLoading || !selectedFile}
                    style={{ padding: '10px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: (isLoading || !selectedFile) ? 'not-allowed' : 'pointer' }}
                >
                    {isLoading ? 'Uploading...' : `Send ${selectedFile?.name || 'file'}`}
                </button>
            </form>

            {previewUrl && (
                <div style={{ marginBottom: '20px', padding: '10px', border: '1px solid #ddd', borderRadius: '4px' }}>
                    <img
                        src={previewUrl}
                        alt="Preview of the selected file"
                        style={{
                            maxWidth: '100%',
                            maxHeight: '200px',
                            objectFit: 'contain',
                            display: 'block',
                            borderRadius: '4px'
                        }}
                    />
                </div>
            )}

            {error && (
                <div style={{ color: 'red', border: '1px solid red', padding: '10px', borderRadius: '4px' }}>
                    Error: {error}
                </div>
            )}

            {responseJson && (
                <div style={{ marginTop: '20px', backgroundColor: '#f4f4f4', padding: '15px', borderRadius: '8px' }}>
                    <h3>✨ Top-10 similar images</h3>

                    {Object.entries(JSON.parse(responseJson.result))
                        .map(([filename, score]) => {
                            const imageUrl = `http://127.0.0.1:8000/images/${filename}`;

                            return (
                                <div
                                    key={filename}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        marginBottom: '10px',
                                        padding: '5px',
                                        borderBottom: '1px solid #ddd'
                                    }}
                                >
                                    <div style={{ display: 'flex', alignItems: 'center' }}>
                                        <img
                                            src={imageUrl}
                                            alt={`Similar image: ${filename}`}
                                            style={{
                                                width: '400px',
                                                height: '400px',
                                                objectFit: 'cover',
                                                marginRight: '15px',
                                                borderRadius: '4px'
                                            }}
                                            onError={(e) => {
                                                const target = e.target as HTMLImageElement;
                                                target.onerror = null; // Предотвращаем бесконечный цикл
                                                target.style.opacity = '0.5';
                                                target.alt = `[Image not found] ${filename}`;
                                                target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60"><rect width="60" height="60" fill="#ccc"/><text x="30" y="35" font-size="10" fill="#666" text-anchor="middle">No Img</text></svg>';
                                            }}
                                        />
                                        <span style={{ fontSize: '14px', color: '#333' }}>
                                {filename}
                            </span>
                                    </div>
                                    <span style={{
                                        fontWeight: 'bold',
                                        fontSize: '16px',
                                        color: '#007bff',
                                        backgroundColor: '#e6f0ff',
                                        padding: '4px 8px',
                                        borderRadius: '12px'
                                    }}>
                            {score as number}
                        </span>
                                </div>
                            );
                        })}
                    <p style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
                        **Uploaded file name:** {responseJson.uploaded_file}
                    </p>
                </div>
            )}
        </div>
    );
};

export default ImageUploader;