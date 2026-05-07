'use client'

interface Researcher {
  id: number
  name: string
  role: string
  email: string
  avatarUrl: string
  avatarAlt: string
}

const researchers: Researcher[] = [
  {
    id: 1,
    name: 'Kristian Bautista',
    role: 'Project Manager',
    email: 'kristiandavidbautista@gmail.com',
    avatarUrl: 'http://localhost:3845/assets/f111a4d9e98c2f1849285d198126666303e67f65.png',
    avatarAlt: '3D Avatar - Kristian David Bautista',
  },
  {
    id: 2,
    name: 'Angel Letada',
    role: 'SUMO Engineer',
    email: 'angel.letada1205@gmail.com',
    avatarUrl: 'http://localhost:3845/assets/eaa320717b7e77fd08d1bdaf9802cc375eb36366.png',
    avatarAlt: '3D Avatar - Angel Letada',
  },
  {
    id: 3,
    name: 'Michael Pascual',
    role: 'SUMO Engineer',
    email: 'michaelkevinpascual47@gmail.com',
    avatarUrl: 'http://localhost:3845/assets/5f8ea6b9caf08d167684ed154ad8a85f97b6913b.png',
    avatarAlt: '3D Avatar - Michael Pascual',
  },
  {
    id: 4,
    name: 'Marianne Santos',
    role: 'Software Engineer',
    email: 'mariannesantos174@gmail.com',
    avatarUrl: 'http://localhost:3845/assets/e61b32a6b96823a8b0214ef17a3aac015a2ed382.png',
    avatarAlt: '3D Avatar - Marianne Santos',
  },
]

export const ResearchersSection = () => {
  return (
    <div className="w-full">
      <h2 className="font-bold text-civiq-purple text-[48px] mb-16 text-center">
        The Researchers
      </h2>

      <div className="grid grid-cols-4 gap-12">
        {researchers.map((researcher) => (
          <div
            key={researcher.id}
            className="flex flex-col items-center"
          >
            {/* Avatar */}
            <div className="relative size-[140px] rounded-full overflow-hidden bg-white shadow-[0px_8px_16px_0px_rgba(0,0,0,0.15)] mb-8 border-4 border-white hover:shadow-[0px_12px_24px_0px_rgba(0,0,0,0.2)] transition-shadow">
              <img
                alt={researcher.avatarAlt}
                className="object-cover size-full"
                src={researcher.avatarUrl}
              />
            </div>

            {/* Card */}
            <div className="bg-[#f6f6f6] rounded-[35px] shadow-[0px_4px_4px_0px_rgba(0,0,0,0.25)] p-8 text-center w-full hover:shadow-[0px_8px_12px_0px_rgba(0,0,0,0.15)] transition-shadow">
              <p className="font-bold text-civiq-dark text-[22px] mb-3">
                {researcher.name}
              </p>
              <p className="font-normal italic text-civiq-dark text-[16px] mb-6">
                {researcher.role}
              </p>
              <a
                href={`mailto:${researcher.email}`}
                className="font-normal text-civiq-dark text-[13px] hover:text-civiq-blue transition-colors whitespace-nowrap text-ellipsis overflow-hidden block"
              >
                {researcher.email}
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
