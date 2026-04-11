'use client'

export const AnimatedBackground = () => {
  // Blobs tuned for a deep-dark background — vivid core, wide soft glow
  const blobs = [
    {
      style: {
        width: 640,
        height: 640,
        background: 'radial-gradient(circle at 38% 38%, #7c3aed 0%, #4c1d95 55%, transparent 100%)',
        top: '-200px',
        left: '-170px',
        animation: 'blob1 14s ease-in-out infinite',
        animationDelay: '0s',
      },
    },
    {
      style: {
        width: 580,
        height: 580,
        background: 'radial-gradient(circle at 60% 38%, #06b6d4 0%, #0e4f6b 55%, transparent 100%)',
        top: '-120px',
        right: '-180px',
        animation: 'blob2 17s ease-in-out infinite',
        animationDelay: '-4s',
      },
    },
    {
      style: {
        width: 520,
        height: 520,
        background: 'radial-gradient(circle at 50% 58%, #db2777 0%, #7d1050 55%, transparent 100%)',
        bottom: '4%',
        left: '12%',
        animation: 'blob3 15s ease-in-out infinite',
        animationDelay: '-7s',
      },
    },
    {
      style: {
        width: 480,
        height: 480,
        background: 'radial-gradient(circle at 44% 54%, #2563eb 0%, #1e3a8a 55%, transparent 100%)',
        bottom: '8%',
        right: '4%',
        animation: 'blob4 18s ease-in-out infinite',
        animationDelay: '-2s',
      },
    },
    {
      style: {
        width: 400,
        height: 400,
        background: 'radial-gradient(circle at 50% 50%, #6d28d9 0%, #3b0764 55%, transparent 100%)',
        top: '40%',
        left: '40%',
        transform: 'translate(-50%, -50%)',
        animation: 'blob5 20s ease-in-out infinite',
        animationDelay: '-10s',
      },
    },
  ]

  return (
    <div
      className="fixed inset-0 overflow-hidden pointer-events-none"
      style={{ zIndex: 0 }}
    >
      {blobs.map((blob, i) => (
        <div
          key={i}
          className="absolute rounded-full"
          style={{
            ...blob.style,
            filter: 'blur(72px)',
            opacity: 0.55,
          }}
        />
      ))}
    </div>
  )
}
